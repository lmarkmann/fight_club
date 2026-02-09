"""Data structures for pose sequences and video keypoints."""

from dataclasses import dataclass

import numpy as np


@dataclass
class VideoKeypoints:
    """Container for extracted pose data from a video."""

    keypoints: np.ndarray   # (T, 17, 3) — x, y, z coordinates
    confidence: np.ndarray  # (T, 17) — per-keypoint confidence
    timestamps: np.ndarray  # (T,) — frame timestamps in seconds
    fps: float
    video_path: str

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0
