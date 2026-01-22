"""Data structures for pose sequences and video keypoints."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseFrame:
    """Pose estimation result for a single video frame."""

    keypoints: np.ndarray  # shape (17, 2) for x, y coordinates
    confidence: np.ndarray  # shape (17,) confidence per keypoint
    frame_index: int
    timestamp_ms: float


@dataclass
class PoseSequence:
    """A temporal sequence of poses representing one action clip."""

    frames: list
    fps: float

    def to_array(self) -> np.ndarray:
        """Convert sequence to numpy array of shape (T, 17, 2) for model input."""
        return np.stack([f.keypoints for f in self.frames], axis=0)

    def get_confidence_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Return boolean mask for keypoints exceeding confidence threshold."""
        confidences = np.stack([f.confidence for f in self.frames], axis=0)
        return confidences >= threshold


@dataclass
class VideoKeypoints:
    """Container for extracted pose data from a video."""

    keypoints: np.ndarray  # shape: (T, 17, 2) - normalized coordinates
    confidence: np.ndarray  # shape: (T, 17) - per-keypoint confidence
    timestamps: np.ndarray  # shape: (T,) - frame timestamps in seconds
    fps: float
    video_path: str

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        if len(self.timestamps) > 1:
            return self.timestamps[-1] - self.timestamps[0]
        return 0.0
