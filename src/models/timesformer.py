from transformers import (
    AutoModelForVideoClassification,
    TimesformerModel,
    VideoMAEImageProcessor,
)

from src.models import ModelConfig, register_model_config


@register_model_config(
    "timesformer",
)
class TimesformerModelConfig(ModelConfig):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model: TimesformerModel = AutoModelForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        videomae_processor = VideoMAEImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        assert isinstance(videomae_processor, VideoMAEImageProcessor)

        self.preprocessor = videomae_processor

    @classmethod
    def get_clip_size(cls):
        return 8

    def get_preprocessor(self):
        return self.preprocessor

    def get_model(self):
        return self.model
