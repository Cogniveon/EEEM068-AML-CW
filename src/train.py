import logging
from typing import Callable

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, Metric
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForVideoClassification

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
    metric: Metric,
):
    """Evaluation step."""
    model.eval()
    inputs, targets = batch
    outputs = model(inputs)
    preds = outputs.logits.argmax(-1)
    result = metric(preds, targets)
    return result.cpu().item()


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

    accelerator = Accelerator()

    processor = AutoImageProcessor.from_pretrained(config.model_name)
    dataset = HMDBSIMPDataset(config.dataset_dir, processor=processor)

    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
        dataset, accelerator, config
    )

    model = AutoModelForVideoClassification.from_pretrained(
        config.model_name,
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
    metric = Accuracy(task="multiclass", num_classes=dataset.num_classes or -1)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if type(loss_fn) == torch.nn.Module:
        loss_fn = loss_fn.to(accelerator.device)
    metric = metric.to(accelerator.device)

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
                for idx, batch in pbar:
                    global_step += 1
                    optimizer.zero_grad()
                    loss, preds = train_step(model, batch, loss_fn, return_preds=True)
                    accelerator.backward(loss)
                    optimizer.step()

                    pbar.set_postfix(
                        {"train/loss": loss.cpu().item(), "lr": scheduler.get_last_lr()}
                    )
                    if idx == 0 and epoch == 0:
                        viz_clip = batch[0].clone().detach().cpu()
                        visualize_predictions(viz_clip, preds, batch[1])
                        viz_clip = visualize_predictions(viz_clip, preds, batch[1])

                        tblogger.add_video(
                            f"train_batch_sample_0",
                            viz_clip,
                            global_step=global_step,
                        )

                    if idx % max(int(len(train_dataloader) / config.log_freq), 1) == 0:
                        tblogger.add_scalar(
                            "train/loss", loss.cpu().item(), global_step=idx
                        )
                        tblogger.add_scalar(
                            "lr", scheduler.get_last_lr(), global_step=idx
                        )

                if epoch % config.eval_freq == 0 or epoch == config.num_epochs - 1:
                    global_step += 1
                    for batch in val_dataloader:
                        accuracy = eval_step(model, batch, metric)
                        pbar.set_postfix(
                            {
                                f"val/{metric.__class__.__name__.lower()}": accuracy,
                                "lr": scheduler.get_last_lr(),
                            }
                        )

                    tblogger.add_scalar(
                        f"val/{metric.__class__.__name__.lower()}",
                        metric.compute().cpu().item(),
                        global_step=global_step,
                    )

                metric.reset()
                scheduler.step()


if __name__ == "__main__":
    main()
