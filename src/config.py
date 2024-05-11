from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from src import models


@dataclass
class Config:
    """Configuration for training a model."""

    model_name: str = "timesformer"
    """See `get_model_config` in `src/models/__init__.py` for available models."""

    dataset_dir: str = "datasets/HMDB_simp"
    log_dir: str = "logs"
    only_eval: bool = False
    only_cpu: bool = False
    checkpoint: Optional[str] = None
    dataset_splits: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    num_epochs: int = 10
    batch_size: int = 4
    seed: int = 6839323
    optimizer: str = "adamw"
    """See `get_optimizer` in `src/utils.py` for available optimizers."""
    optimizer_kwargs: dict = field(
        default_factory=lambda: {"lr": 1e-4, "weight_decay": 1e-2}
    )
    lr_scheduler: str = "onecycle"
    """See `get_lr_scheduler` in `src/utils.py` for available schedulers."""
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {})
    loss_fn: str = field(default="cross_entropy")
    """See `torch.nn.functional` for available loss functions"""
    log_freq: int = 10  # Number of times to log per epoch
    eval_freq: int = 5  # Frequency of the evaluation step
