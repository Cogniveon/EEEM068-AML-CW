import logging
import os
from typing import Callable

import torch
import torchmetrics
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, ConfusionMatrix, Metric
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForVideoClassification,
    VideoMAEImageProcessor,
    VivitImageProcessor,
)

from src import utils
from src.dataset import HMDBSIMPDataset
from src.utils import visualize_predictions

log = logging.getLogger(__name__)


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: torch.nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    return_preds: bool = False,
):
    """Train step."""
    model.train()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs.logits, targets)

    if return_preds:
        preds = outputs.logits.argmax(-1)
        return loss, preds
    return loss


def eval_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: torch.nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    accuracy: Metric,
    confusion_matrix: Metric,
):
    """Evaluation step."""
    model.eval()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs.logits, targets)
    preds = outputs.logits.argmax(-1)
    return (
        loss,
        accuracy.update(preds, targets).item(),
        confusion_matrix.update(preds, targets),
    )


def prepare_dataloaders(
    dataset: HMDBSIMPDataset,
    accelerator: Accelerator,
    config: DictConfig | ListConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare dataloaders."""
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, config.dataset_splits[:3]
    )

    train_dataloader = accelerator.prepare_data_loader(
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
    )

    val_dataloader = accelerator.prepare_data_loader(
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
    )

    test_dataloader = accelerator.prepare_data_loader(
        DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
    )

    return train_dataloader, val_dataloader, test_dataloader


def main(config: ListConfig | DictConfig | None = None):
    """Main function."""

    if config is None:
        config = utils.get_config()

    utils.setup_logging(config)
    set_seed(config.seed)

    tblogger = SummaryWriter(config.log_dir)
    tblogger.add_text(
        "hparams",
        """```yaml
{yaml}
```
""".format(
            yaml=OmegaConf.to_yaml(config, resolve=True)
        ),
        global_step=0,
    )
    log.info(f"Configuration: {OmegaConf.to_container(config, resolve=True)}")

    accelerator = Accelerator(cpu=True if config.only_cpu else False)

    if "vivit" in config.model_name:
        processor = VivitImageProcessor.from_pretrained(config.model_name)
    elif "videomae" in config.model_name:
        processor = VideoMAEImageProcessor.from_pretrained(config.model_name)
    elif "timesformer" in config.model_name:
        processor = VideoMAEImageProcessor.from_pretrained(config.model_name)
    else:
        processor = AutoImageProcessor.from_pretrained(config.model_name)
    dataset = HMDBSIMPDataset(
        config.dataset_dir,
        processor=processor,  # type: ignore
        clip_size=config.clip_size,
    )

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        dataset, accelerator, config
    )

    model = AutoModelForVideoClassification.from_pretrained(
        config.model_name,
        num_labels=dataset.num_classes,
        ignore_mismatched_sizes=True,
    )

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_dataloader),
    )
    loss_fn = torch.nn.functional.cross_entropy
    accuracy = Accuracy(
        task="multiclass",
        num_classes=dataset.num_classes or -1,
    ).to(accelerator.device)
    confusion_matrix = ConfusionMatrix(
        task="multiclass", num_classes=dataset.num_classes or -1
    ).to(accelerator.device)
    train_loss = torchmetrics.MeanMetric().to(accelerator.device)
    val_loss = torchmetrics.MeanMetric().to(accelerator.device)
    test_loss = torchmetrics.MeanMetric().to(accelerator.device)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if type(loss_fn) == torch.nn.Module:
        loss_fn = loss_fn.to(accelerator.device)

    if config.checkpoint is not None:
        accelerator.load_state(config.checkpoint)

    if not config.only_eval:
        global_step = -1
        for epoch in range(config.num_epochs):
            with tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}",
            ) as pbar:
                random_index = torch.randint(0, len(train_dataloader), (1,)).item()
                for idx, batch in pbar:
                    global_step += 1
                    optimizer.zero_grad()
                    loss, preds = train_step(model, batch, loss_fn, return_preds=True)
                    accelerator.backward(loss)
                    train_loss.update(loss.item())
                    optimizer.step()
                    scheduler.step()

                    pbar.set_postfix(
                        {
                            "train/loss": loss.cpu().item(),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                    if idx == random_index:
                        viz_clip = batch[0].clone().detach()
                        preds = model(viz_clip).logits.argmax(-1)
                        viz_clip = visualize_predictions(viz_clip, preds, batch[1])

                        tblogger.add_video(
                            f"train_batch_sample",
                            viz_clip,
                            global_step=epoch,
                        )

                    if idx % max(int(len(train_dataloader) / config.log_freq), 1) == 0:
                        tblogger.add_scalar(
                            "train/loss", train_loss.compute(), global_step=idx
                        )
                        train_loss.reset()
                        tblogger.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step=global_step
                        )
                        tblogger.flush()

            if epoch % config.eval_freq == 0 or epoch == config.num_epochs - 1:
                with tqdm(
                    enumerate(val_dataloader),
                    total=len(val_dataloader),
                    desc=f"Validating {epoch + 1}",
                ) as pbar:
                    random_index = torch.randint(0, len(val_dataloader), (1,)).item()
                    for idx, batch in pbar:
                        global_step += 1
                        loss, acc, confmat = eval_step(
                            model, batch, loss_fn, accuracy, confusion_matrix
                        )
                        val_loss.update(loss.item())
                        pbar.set_postfix(
                            {
                                "val/accuracy": acc,
                            }
                        )

                        if idx == random_index:
                            viz_clip = batch[0].clone().detach()
                            preds = model(viz_clip).logits.argmax(-1)
                            viz_clip = visualize_predictions(viz_clip, preds, batch[1])

                            tblogger.add_video(
                                f"val_batch_sample",
                                viz_clip,
                                global_step=epoch,
                            )

                        if (
                            idx % max(int(len(train_dataloader) / config.log_freq), 1)
                            == 0
                        ):
                            tblogger.add_scalar(
                                f"val/accuracy",
                                accuracy.compute().cpu().item(),
                                global_step=global_step,
                            )
                            accuracy.reset()

                            tblogger.add_scalar(
                                "val/loss", val_loss.compute(), global_step=idx
                            )
                            val_loss.reset()
                            tblogger.flush()

                    # Plot confusion matrix at the end of validation
                    tblogger.add_figure(
                        f"val/confusion_matrix",
                        confusion_matrix.plot(add_text=False)[0],
                        global_step=global_step,
                    )
                    confusion_matrix.reset()

                accelerator.save_state(
                    os.path.join(str(tblogger.logdir), "checkpoints", f"epoch_{epoch}")
                )

    accuracy.reset()
    confusion_matrix.reset()
    with tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc=f"Testing",
    ) as pbar:
        random_index = torch.randint(0, len(test_dataloader), (1,)).item()
        for idx, batch in pbar:
            loss, acc, confmat = eval_step(
                model, batch, loss_fn, accuracy, confusion_matrix
            )
            test_loss.update(loss.item())
            pbar.set_postfix(
                {
                    "test/accuracy": acc,
                }
            )

            if idx == random_index:
                viz_clip = batch[0].clone().detach()
                preds = model(viz_clip).logits.argmax(-1)
                viz_clip = visualize_predictions(viz_clip, preds, batch[1])

                tblogger.add_video(
                    f"test_batch_sample",
                    viz_clip,
                    global_step=0,
                )

            tblogger.add_scalar(
                f"test/accuracy",
                accuracy.compute().cpu().item(),
                global_step=idx,
            )
            accuracy.reset()

            tblogger.add_scalar("test/loss", test_loss.compute(), global_step=idx)
            test_loss.reset()

        tblogger.add_figure(
            f"test/confusion_matrix",
            confusion_matrix.plot(add_text=False)[0],
            global_step=0,
        )
        confusion_matrix.reset()

if __name__ == "__main__":
    main()
