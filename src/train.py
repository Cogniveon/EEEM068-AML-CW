import logging
import os
import sys
import warnings
from typing import Callable, Literal

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from rich.logging import RichHandler
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, Metric
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from src.config import Config
from src.dataset import HMDBSIMPDataset
from src.utils import visualize_predictions

log = logging.getLogger(__name__)


def common_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    accelerator: Accelerator,
    loss_fn: torch.nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metric: Metric,
):
    """Common step for both training and validation."""
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs.logits, targets)
    accelerator.backward(loss)

    preds = outputs.logits.argmax(-1)
    accuracy = metric(preds, targets)

    return loss, accuracy, preds


def step_epoch(
    accelerator: Accelerator,
    epoch: int,
    total_epochs: int,
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    loss_fn: torch.nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    logger: SummaryWriter,
    log_freq: int,
    mode: Literal["train", "test", "val"] = "train",
) -> None:
    """Train one epoch."""
    log.debug(f"Starting {mode} epoch {epoch + 1}/{total_epochs}")
    # log.debug(f"{log_freq=}")
    # log.debug(f"{len(dataloader)=}")
    # log.debug(f"{int(len(dataloader) / log_freq)=}")
    log_step = max(int(len(dataloader) / log_freq), 1)
    log.debug(f"Logging every {log_step} steps")

    if mode == "train":
        model.train()
    else:
        model.eval()

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=(
            f"Epoch {epoch + 1:3.0f}/{total_epochs}"
            if mode == "train"
            else (f"Validating" if mode == "val" else "Testing")
        ),
        position=0,
    ) as progress:
        random_index = torch.randint(0, len(dataloader), (1,)).item()
        for idx, batch in progress:
            optimizer.zero_grad()
            loss, accuracy, preds = common_step(
                model, batch, accelerator, loss_fn, metric
            )
            optimizer.step()

            # Once every epoch, log the predictions
            if idx == random_index:
                correct_indices = (preds == batch[1]).nonzero().squeeze()
                incorrect_indices = (preds != batch[1]).nonzero().squeeze()
                log.debug(f"Correct indices: {correct_indices}")
                log.debug(f"Incorrect indices: {incorrect_indices}")

                # Add a red/green border to images
                viz_clip = batch[0].clone().detach().cpu()
                visualize_predictions(viz_clip, preds, batch[1])
                viz_clip = visualize_predictions(viz_clip, preds, batch[1])

                logger.add_video(
                    f"{mode}_batch",
                    viz_clip,
                    global_step=epoch,
                )

            progress.set_postfix(
                {"loss": loss.cpu().item(), "accuracy": accuracy.cpu().item()}
            )
            if idx % log_step == 0:
                logger.add_scalars(
                    "loss",
                    {
                        mode: loss.cpu().item(),
                    },
                    global_step=idx,
                )
                logger.add_scalars(
                    "accuracy",
                    {
                        mode: accuracy.cpu().item(),
                    },
                    global_step=idx,
                )

                log.debug(
                    f"Epoch {epoch + 1:3.0f}/{total_epochs} {mode} step {idx}/{len(dataloader)}: loss={loss.cpu().item()}, accuracy={accuracy.cpu().item()}"
                )


if __name__ == "__main__":
    rootLogger = logging.getLogger()

    for handler in rootLogger.handlers:
        rootLogger.removeHandler(handler)

    rootLogger.addHandler(
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            omit_repeated_times=False,
        )
    )
    rootLogger.setLevel(logging.INFO)
    logging.getLogger("src").setLevel(logging.DEBUG)
    logging.getLogger("src.dataset").setLevel(logging.INFO)
    log.setLevel(logging.DEBUG)
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

    config = OmegaConf.structured(Config)
    cli_params = OmegaConf.from_cli(sys.argv[1:])

    config = OmegaConf.merge(config, cli_params)

    set_seed(config.seed)

    tblogger = SummaryWriter(config.log_dir)
    tblogger.add_text("hparams", OmegaConf.to_yaml(config), global_step=0)
    log.info(f"Configuration: {config}")

    accelerator = Accelerator(
        project_dir=config.log_dir,
    )

    processor = AutoImageProcessor.from_pretrained(config.model_name)
    model = AutoModelForVideoClassification.from_pretrained(
        config.model_name,
    )

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    dataset = HMDBSIMPDataset(config.dataset_dir, transform=processor)

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
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    loss_fn = torch.nn.functional.cross_entropy
    metric = Accuracy(task="multiclass", num_classes=dataset.num_classes or -1)

    model, optimizer, train_dataloader, scheduler, metric = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler, metric
    )
    if type(loss_fn) == torch.nn.Module:
        loss_fn = accelerator.prepare(loss_fn)

    accelerator.register_for_checkpointing(scheduler)

    for epoch in range(config.num_epochs):
        metric.reset()
        for stage in ["train", "val"]:
            metric.reset()
            step_epoch(
                accelerator=accelerator,
                epoch=epoch,
                total_epochs=config.num_epochs,
                model=model,
                dataloader=train_dataloader if stage == "train" else val_dataloader,
                optimizer=optimizer,
                metric=metric,
                loss_fn=loss_fn,
                logger=tblogger,
                mode=stage,  # type: ignore
                log_freq=config.log_freq,
            )

        scheduler.step()

        # Checkpoint
        output_dir = os.path.join(config.log_dir, f"epoch_{epoch}")
        accelerator.save_state(output_dir)

    # Test
    metric.reset()
    step_epoch(
        accelerator=accelerator,
        epoch=0,
        total_epochs=1,
        model=model,
        dataloader=test_dataloader,
        optimizer=optimizer,
        metric=metric,
        loss_fn=loss_fn,
        logger=tblogger,
        mode="test",
        log_freq=config.log_freq,
    )

    tblogger.close()
