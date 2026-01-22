"""Pose normalization utilities for scale and position invariance."""

import numpy as np

from .constants import COCO_KEYPOINTS


def normalize_pose(
    keypoints: np.ndarray,
    reference_joint1: int = COCO_KEYPOINTS["left_hip"],
    reference_joint2: int = COCO_KEYPOINTS["right_hip"],
) -> np.ndarray:
    """
    Normalize poses to be invariant to camera distance and subject position.

    The hip midpoint becomes the origin, and the hip width becomes the unit
    of measurement. This makes the representation robust to where the subject
    stands in the frame and how far away the camera is.

    Rotation is preserved because stance orientation carries semantic
    information: an orthodox fighter's cross comes from a different side
    than a southpaw's.
    """
    hip1 = keypoints[..., reference_joint1, :]
    hip2 = keypoints[..., reference_joint2, :]
    origin = (hip1 + hip2) / 2

    hip_distance = np.linalg.norm(hip1 - hip2, axis=-1, keepdims=True)
    hip_distance = np.maximum(hip_distance, 1e-6)  # prevent division by zero

    centered = keypoints - origin[..., np.newaxis, :]
    normalized = centered / hip_distance[..., np.newaxis]

    return normalized
