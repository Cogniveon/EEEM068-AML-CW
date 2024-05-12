import logging
import os
from typing import Callable, Optional

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

from src import models, utils
from src.dataset import HMDBSIMPDataset

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
    accuracy5: Metric,
    confusion_matrix: Optional[Metric] = None,
):
    """Evaluation step."""
    model.eval()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs.logits, targets)
    preds = outputs.logits.argmax(-1)
    accuracy.update(preds.clone().detach().cpu(), targets.clone().detach().cpu())
    accuracy5.update(
        torch.nn.functional.softmax(outputs.logits, dim=-1).clone().detach().cpu(),
        targets.clone().detach().cpu(),
    )
    if confusion_matrix is not None:
        confusion_matrix.update(
            preds.clone().detach().cpu(), targets.clone().detach().cpu()
        )
    return loss, accuracy.compute(), accuracy5.compute()


def prepare_dataloaders(
    dataset: HMDBSIMPDataset,
    accelerator: Accelerator,
    batch_size: int,
    dataset_splits: list[float] = [0.8, 0.1, 0.1],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare dataloaders."""
    log.info(f"Preparing dataloaders with splits: {dataset_splits}")
    train_dataset, val_dataset, test_dataset = random_split(dataset, dataset_splits)[:3]

    train_dataloader = accelerator.prepare_data_loader(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
    )

    val_dataloader = accelerator.prepare_data_loader(
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    )

    test_dataloader = accelerator.prepare_data_loader(
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    )

    return train_dataloader, val_dataloader, test_dataloader


def main(config: ListConfig | DictConfig | None = None):
    """Main function."""

    if config is None:
        config = utils.get_config()

    utils.setup_logging(config)

    log.info(f"Seed: {config.seed}")
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
    log.info(f"Using device: {accelerator.device}")

    dataset = HMDBSIMPDataset(
        config.dataset_dir,
        clip_size=models.get_model_config(config.model_name).get_clip_size(),
    )

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        dataset, accelerator, config.batch_size, config.dataset_splits
    )

    model_config = models.get_model_config(config.model_name)(
        num_classes=dataset.num_classes
    )

    preprocessor = model_config.get_preprocessor()
    dataset.set_preprocessor(preprocessor)
    tblogger.add_text(
        "preprocessor",
        """{preprocessor}""".format(preprocessor=preprocessor),
        global_step=0,
    )

    log.info(f"Fetching model config: {config.model_name}")
    model = model_config.get_model()

    log.info(
        f"Using optimizer: {config.optimizer} with kwargs: {config.optimizer_kwargs}"
    )
    optimizer = utils.get_optimizer(model, config.optimizer, config.optimizer_kwargs)

    lr_scheduler_kwargs = config.lr_scheduler_kwargs
    if config.lr_scheduler == "onecycle":
        lr_scheduler_kwargs.setdefault("max_lr", config.optimizer_kwargs["lr"])
        lr_scheduler_kwargs.setdefault("steps_per_epoch", len(train_dataloader))
        lr_scheduler_kwargs.setdefault("epochs", config.num_epochs)
    log.info(
        f"Using lr_scheduler: {config.lr_scheduler} with kwargs: {lr_scheduler_kwargs}"
    )
    scheduler = utils.get_lr_scheduler(
        optimizer,
        config.lr_scheduler,
        **lr_scheduler_kwargs,
    )

    log.info(f"Using loss function: {config.loss_fn}")
    loss_fn = getattr(torch.nn.functional, config.loss_fn)

    accuracy = Accuracy(
        task="multiclass",
        num_classes=dataset.num_classes or -1,
    )
    accuracy5 = Accuracy(
        task="multiclass",
        top_k=5,
        num_classes=dataset.num_classes or -1,
    )
    train_loss = torchmetrics.MeanMetric()
    val_loss = torchmetrics.MeanMetric()
    test_loss = torchmetrics.MeanMetric()

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if type(loss_fn) == torch.nn.Module:
        loss_fn = loss_fn.to(accelerator.device)

    if config.checkpoint is not None:
        accelerator.load_state(config.checkpoint)

    # visualizer = utils.VisualizeAttention(model)
    # visualizer.visualize_attention(torch.randn(1, 8, 3, 224, 224))

    if not config.only_eval:
        global_step = -1
        for epoch in range(config.num_epochs):
            with tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Train {epoch + 1}",
            ) as pbar:
                random_index = torch.randint(0, len(train_dataloader), (1,)).item()
                for idx, batch in pbar:
                    global_step += 1
                    optimizer.zero_grad()
                    loss, preds = train_step(model, batch, loss_fn, return_preds=True)
                    accelerator.backward(loss)
                    train_loss.update(loss.cpu().item())
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
                        viz_clip = utils.visualize_predictions(
                            viz_clip, preds, batch[1]
                        )

                        tblogger.add_video(
                            f"train_batch_sample",
                            viz_clip,
                            global_step=epoch,
                        )

                    if idx % max(int(len(train_dataloader) / config.log_freq), 1) == 0:
                        tblogger.add_scalar(
                            "train/loss",
                            train_loss.compute(),
                            global_step=global_step,
                        )
                        train_loss.reset()
                        tblogger.add_scalar(
                            "lr",
                            scheduler.get_last_lr()[0],
                            global_step=global_step,
                        )
                        tblogger.flush()

                log.info(f"Saving checkpoint for epoch: {epoch}")
                accelerator.save_state(
                    os.path.join(str(tblogger.logdir), "checkpoints", f"epoch_{epoch}")
                )

            if epoch % config.eval_freq == 0 or epoch == config.num_epochs - 1:
                with tqdm(
                    enumerate(val_dataloader),
                    total=len(val_dataloader),
                    desc=f"Val {epoch + 1}",
                ) as pbar:
                    random_index = torch.randint(0, len(val_dataloader), (1,)).item()
                    for idx, batch in pbar:
                        torch.cuda.empty_cache()
                        global_step += 1
                        loss, acc, acc5 = eval_step(
                            model, batch, loss_fn, accuracy, accuracy5
                        )
                        val_loss.update(loss.cpu().item())
                        pbar.set_postfix(
                            {
                                "val/loss": val_loss.compute().item(),
                                "val/acc": acc.item(),
                                "val/acc5": acc5.item(),
                            }
                        )

                        if idx == random_index:
                            viz_clip = batch[0].clone().detach()
                            preds = model(viz_clip).logits.argmax(-1)
                            viz_clip = utils.visualize_predictions(
                                viz_clip, preds, batch[1]
                            )

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
                                f"val/acc",
                                acc.item(),
                                global_step=global_step,
                            )
                            tblogger.add_scalar(
                                f"val/acc5",
                                acc5.item(),
                                global_step=global_step,
                            )

                            tblogger.add_scalar(
                                "val/loss",
                                val_loss.compute(),
                                global_step=global_step,
                            )
                            val_loss.reset()
                            tblogger.flush()

                    log.info(f"Val Accuracy: {accuracy.compute().item()}")
                    log.info(f"Val Accuracy@5: {accuracy5.compute().item()}")
                    accuracy.reset()
                    accuracy5.reset()

    torch.cuda.empty_cache()
    confusion_matrix = ConfusionMatrix(
        task="multiclass", num_classes=dataset.num_classes or -1
    )

    with tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc=f"Testing",
    ) as pbar:
        random_index = torch.randint(0, len(test_dataloader), (1,)).item()
        for idx, batch in pbar:
            loss, acc, acc5 = eval_step(
                model, batch, loss_fn, accuracy, accuracy5, confusion_matrix
            )
            test_loss.update(loss.cpu().item())
            pbar.set_postfix(
                {
                    "test/loss": test_loss.compute().item(),
                    "test/acc": acc.item(),
                    "test/acc5": acc5.item(),
                }
            )
            tblogger.add_scalar(
                f"test/acc",
                acc.item(),
                global_step=idx,
            )
            tblogger.add_scalar(
                f"test/acc5",
                acc5.item(),
                global_step=idx,
            )
            tblogger.add_scalar(
                "test/loss", test_loss.compute().item(), global_step=idx
            )

            if idx == random_index:
                viz_clip = batch[0].clone().detach()
                preds = model(viz_clip).logits.argmax(-1)
                viz_clip = utils.visualize_predictions(viz_clip, preds, batch[1])

                tblogger.add_video(
                    f"test_batch_sample",
                    viz_clip,
                    global_step=0,
                )

        tblogger.add_figure(
            f"test/confusion_matrix",
            confusion_matrix.plot(add_text=False)[0],
            global_step=0,
        )
        tblogger.add_text(
            "metrics",
            """Test Accuracy: {acc:0.2f}
Test Accuracy@5: {acc5:0.2f}
Test Loss: {loss:0.2f}""".format(
                acc=acc.item(),
                acc5=acc5.item(),
                loss=test_loss.compute().item(),
            ),
            global_step=0,
        )
        tblogger.flush()
        log.info(f"Test Accuracy: {accuracy.compute().item()}")
        log.info(f"Test Accuracy@5: {accuracy5.compute().item()}")
        log.info(f"Test Loss: {test_loss.compute().item()}")

if __name__ == "__main__":
    main()
