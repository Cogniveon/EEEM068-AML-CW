import logging
import sys
import warnings

import numpy as np
import torch
import transformers
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.logging import RichHandler

from src.config import Config

log = logging.getLogger(__name__)

def draw_box(video: torch.Tensor, border_size=5, color="red"):
    """
    Draw a rectangle with the given coordinates and edge color on the provided axes object.
    :param ax: Matplotlib axes object.
    :param coord: Tuple of (x, y, width, height) for the rectangle.
    :param edge_color: Color of the rectangle edge.
    :param label: Label text to display with the rectangle.
    """
    video[:, :, :border_size, :] = 0
    video[:, :, -border_size:, :] = 0
    video[:, :, :, :border_size] = 0
    video[:, :, :, -border_size:] = 0

    if color == "red":
        video[:, 0, :border_size, :] = 1
        video[:, 0, -border_size:, :] = 1
        video[:, 0, :, :border_size] = 1
        video[:, 0, :, -border_size:] = 1

    elif color == "green":
        video[:, 1, :border_size, :] = 1
        video[:, 1, -border_size:, :] = 1
        video[:, 1, :, :border_size] = 1
        video[:, 1, :, -border_size:] = 1

    elif color == "blue":
        video[:, 2, :border_size, :] = 1
        video[:, 2, -border_size:, :] = 1
        video[:, 2, :, :border_size] = 1
        video[:, 2, :, -border_size:] = 1

    return video


def visualize_predictions(video_tensor, predictions, labels):
    """
    Visualize predictions for a batch of videos at a specified frame index.
    :param video_tensor: Tensor of shape [B, T, C, H, W].
    :param predictions: List of predictions for each batch item.
    :param labels: List of true labels for each batch item.
    :param frame_index: Frame index to visualize within each video.
    """
    batch_size = video_tensor.shape[0]

    count_correct = 0
    for batch_idx in range(batch_size):
        # Determine box color based on prediction correctness
        color = "green" if predictions[batch_idx] == labels[batch_idx] else "red"
        count_correct += 1 if predictions[batch_idx] == labels[batch_idx] else 0
        # label = f"Pred: {predictions[i]}, GT: {labels[i]}"
        video_tensor[batch_idx] = draw_box(video_tensor[batch_idx], color=color)

    log.info(f"Visualizing predictions: {count_correct}/{batch_size} correct...")
    return video_tensor


def setup_logging(config: ListConfig | DictConfig):
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
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="moviepy")
    transformers.logging.set_verbosity_error()


def get_config() -> ListConfig | DictConfig:
    cli_params = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.merge(OmegaConf.structured(Config), cli_params)
    return config


def get_optimizer(model: torch.nn.Module, optimizer_name: str, optimizer_kwargs: dict):
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not found in {optimizers.keys()}")
    return optimizers[optimizer_name](model.parameters(), **optimizer_kwargs)


def get_lr_scheduler(optimizer, scheduler_name: str, **kwargs):
    schedulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "multistep": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "cyclic": torch.optim.lr_scheduler.CyclicLR,
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    }

    if scheduler_name not in schedulers:
        raise ValueError(f"Scheduler {scheduler_name} not found in {schedulers.keys()}")
    return schedulers[scheduler_name](optimizer, **kwargs, last_epoch=-1)


class VisualizeAttention:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.hooks = []

    def get_attn_t(self, module, input, output):
        self.time_attentions.append(output.detach().cpu())

    def get_attn_s(self, module, input, output):
        self.space_attentions.append(output.detach().cpu())

    def combine_divided_attention(self, attn_t, attn_s):
        pass

    def visualize_attention(self, inputs: torch.Tensor):
        self.time_attentions = []
        self.space_attentions = []
        self.attentions = []

        for name, m in self.model.named_modules():
            if "temporal_attn.attn_drop" in name:
                self.hooks.append(m.register_forward_hook(self.get_attn_t))
            elif "attn.attn_drop" in name:
                self.hooks.append(m.register_forward_hook(self.get_attn_s))

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)

        for h in self.hooks:
            h.remove()

        for attn_t, attn_s in zip(self.time_attentions, self.space_attentions):
            self.attentions.append(self.combine_divided_attention(attn_t, attn_s))

        p, t = self.attentions[0].shape[0], self.attentions[0].shape[1]
        result = torch.eye(p * t)
        for attention in self.attentions:
            attention = rearrange(attention, "p1 t1 p2 t2 -> (p1 t1) (p2 t2)")
            result = torch.matmul(attention, result)
        mask = rearrange(result, "(p1 t1) (p2 t2) -> p1 t1 p2 t2", p1=p, p2=p)
        mask = mask.mean(dim=1)
        mask = mask[0, 1:, :]
        width = int(mask.size(0) ** 0.5)
        mask = rearrange(mask, "(h w) t -> h w t", w=width).numpy()
        mask = mask / np.max(mask)
        # image_masks = self.create_masks(list(rearrange(masks, 'h w t -> t h w')),np_imgs)
