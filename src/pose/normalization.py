"""Pose normalization and motion feature computation."""

import numpy as np

from .constants import COCO_KEYPOINTS


def normalize_pose(
    keypoints: np.ndarray,
    reference_joint1: int = COCO_KEYPOINTS["left_hip"],
    reference_joint2: int = COCO_KEYPOINTS["right_hip"],
) -> np.ndarray:
    """Normalize poses: hip-centered, hip-width-scaled. Preserves rotation."""
    hip1 = keypoints[..., reference_joint1, :]
    hip2 = keypoints[..., reference_joint2, :]
    origin = (hip1 + hip2) / 2
    scale = np.maximum(np.linalg.norm(hip1 - hip2, axis=-1, keepdims=True), 1e-6)
    return (keypoints - origin[..., np.newaxis, :]) / scale[..., np.newaxis]


def compute_motion_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute velocity and acceleration, concatenate with position.

    Input:  (T, 17, C) — position coordinates
    Output: (T, 17, 3*C) — [position, velocity, acceleration]
    """
    velocity = np.zeros_like(keypoints)
    velocity[1:] = keypoints[1:] - keypoints[:-1]

    acceleration = np.zeros_like(keypoints)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([keypoints, velocity, acceleration], axis=-1)
