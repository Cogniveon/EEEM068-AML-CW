MODEL_ZOO = {}


def register_model_config(name: str):
    def decorator(cls):
        MODEL_ZOO[name] = cls
        return cls

    return decorator


class ModelConfig:
    def __init__(self, num_classes: int) -> None:
        pass

    def get_preprocessor(self):
        raise NotImplementedError

    @classmethod
    def get_clip_size(cls) -> int:
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError


def get_model_config(name: str) -> type[ModelConfig]:
    return MODEL_ZOO[name]


# Import model configurations here to register them
from . import timesformer, videomae, vivit
