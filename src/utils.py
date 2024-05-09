import logging
import sys
import warnings

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.logging import RichHandler

from src.config import Config


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

    for batch_idx in range(batch_size):
        # Determine box color based on prediction correctness
        color = "green" if predictions[batch_idx] == labels[batch_idx] else "red"
        # label = f"Pred: {predictions[i]}, GT: {labels[i]}"
        video_tensor[batch_idx] = draw_box(video_tensor[batch_idx], color=color)

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


def get_config() -> ListConfig | DictConfig:
    config = OmegaConf.structured(Config)
    cli_params = OmegaConf.from_cli(sys.argv[1:])

    config = OmegaConf.merge(config, cli_params)
    return config
