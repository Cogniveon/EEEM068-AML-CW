from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for training a model."""

    model_name: str = "facebook/timesformer-base-finetuned-k400"
    dataset_dir: str = "datasets/HMDB_simp"
    log_dir: str = "logs"
    dataset_splits: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    num_epochs: int = 10
    batch_size: int = 4
    seed: int = 6839323
    lr: float = 1e-4
    weight_decay: float = 1e-2
    log_freq: int = 10  # Number of times to log per epoch
