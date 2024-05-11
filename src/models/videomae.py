from transformers import (
    AutoModelForVideoClassification,
    VideoMAEImageProcessor,
    VideoMAEModel,
)

from src.models import ModelConfig, register_model_config


@register_model_config(
    "videomae",
)
class VideoMAEModelConfig(ModelConfig):
    def __init__(self, num_classes: int):
        self.model: VideoMAEModel = AutoModelForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        videomae_processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        assert isinstance(videomae_processor, VideoMAEImageProcessor)

        self.preprocessor = videomae_processor

    @classmethod
    def get_clip_size(cls):
        return 16

    def get_preprocessor(self):
        return self.preprocessor

    def get_model(self):
        return self.model
