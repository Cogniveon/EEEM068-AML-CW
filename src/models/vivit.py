from transformers import (
    AutoModelForVideoClassification,
    VivitImageProcessor,
    VivitModel,
)

from src.models import ModelConfig, register_model_config


@register_model_config(
    "vivit",
)
class VivitModelConfig(ModelConfig):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model: VivitModel = AutoModelForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        vivit_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400"
        )
        assert isinstance(vivit_processor, VivitImageProcessor)

        self.preprocessor = vivit_processor

    @classmethod
    def get_clip_size(cls):
        return 32

    def get_preprocessor(self):
        return self.preprocessor

    def get_model(self):
        return self.model
