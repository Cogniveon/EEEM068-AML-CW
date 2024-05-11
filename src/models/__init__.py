import logging

log = logging.getLogger(__name__)

MODEL_ZOO = {}


def register_model_config(name: str):
    def decorator(cls):
        MODEL_ZOO[name] = cls
        return cls

    return decorator


class ModelConfig:
    def __init__(self, num_classes: int) -> None:
        log.info(f"Initializing model config with {num_classes} classes")
        self.num_classes = num_classes

    def get_preprocessor(self):
        raise NotImplementedError

    @classmethod
    def get_clip_size(cls) -> int:
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError


def get_model_config(name: str) -> type[ModelConfig]:
    if name not in MODEL_ZOO:
        raise ValueError(f"Model config {name} not found in {MODEL_ZOO.keys()}")

    return MODEL_ZOO[name]


# Import model configurations here to register them
from . import timesformer, videomae, vivit
