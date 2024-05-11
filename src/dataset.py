import logging
import os
from functools import cached_property
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor, VivitImageProcessor

log = logging.getLogger(__name__)


class HMDBSIMPDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __init__(
        self,
        path: str | os.PathLike,
        clip_size: int = 8,
    ):
        self._clip_size = clip_size

        log.info("Initializing HMDB_simp dataset")
        log.info(f"Number of classes: {self.num_classes}")
        log.info(f"Using clip size: {clip_size}")
        
        self.dataset: list[tuple[np.ndarray, int]] = []
        for label in os.listdir(path):
            log.debug(f"Processing label {label}: {os.path.join(path, label)}")
            for video_folder in os.listdir(label_folder := os.path.join(path, label)):
                log.debug(
                    f"Processing video {video_folder}: {os.path.join(label_folder, video_folder)}"
                )
                for idx, clip in self.process_video(
                    os.path.join(label_folder, video_folder), self._clip_size
                ):
                    log.debug(
                        f"Adding video clip {video_folder}/{idx}; frames: {idx * 8}-{(idx + 1) * 8}"
                    )
                    self.dataset.append((clip, self.label2id(label)))

    def process_video(self, video_folder: str, clip_size: int = 8):
        """Reads a video and returns a list of clips of {clip_size} frames each."""
        images = []
        for image in sorted(os.listdir(video_folder)):
            images.append(os.path.join(video_folder, image))

        video = np.array(images)

        # Split the video into clips of {clip_size} frames each leaving the last frames
        # if the number of frames is not divisible by {clip_size}
        video = video[: len(video) - (len(video) % clip_size)]
        video = video.reshape(-1, clip_size)

        for idx, clip in enumerate(video):
            yield idx, clip

    def __len__(self):
        return len(self.dataset)

    def set_preprocessor(
        self,
        preprocessor: Optional[VideoMAEImageProcessor | VivitImageProcessor] = None,
    ):
        self._preprocessor = preprocessor

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        images, label = self.dataset[idx]

        pil_images = []
        for path in images:
            image = Image.open(path)
            pil_images.append(image)

        if self._preprocessor != None:
            image_transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToImage(),
                    T.ToDtype(torch.float, scale=True),
                ]
            )
            tensors = self._preprocessor.preprocess(pil_images).pixel_values
            tensors = torch.stack([image_transform(image) for image in pil_images])

        return tensors, torch.tensor(label)

    @cached_property
    def num_classes(self) -> int:
        return len(self.labels)

    @cached_property
    def labels(self) -> list[str]:
        return [
            "brush_hair",
            "cartwheel",
            "catch",
            "chew",
            "climb",
            "climb_stairs",
            "draw_sword",
            "eat",
            "fencing",
            "flic_flac",
            "golf",
            "handstand",
            "kiss",
            "pick",
            "pour",
            "pullup",
            "pushup",
            "ride_bike",
            "shoot_bow",
            "shoot_gun",
            "situp",
            "smile",
            "smoke",
            "throw",
            "wave",
        ]

    def id2label(self, id: int) -> str:
        return self.labels[id]

    def label2id(self, label: str) -> int:
        return self.labels.index(label)
