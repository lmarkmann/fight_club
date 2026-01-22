"""Pose extraction module for MediaPipe-based keypoint detection."""

from .constants import (
    COCO_KEYPOINTS,
    BOXING_KEYPOINTS,
    MEDIAPIPE_TO_COCO,
    MEDIAPIPE_LANDMARKS,
)
from .dataclasses import PoseFrame, PoseSequence, VideoKeypoints
from .normalization import normalize_pose
from .extraction import (
    extract_video_keypoints,
    save_keypoints,
    load_keypoints,
)

__all__ = [
    "COCO_KEYPOINTS",
    "BOXING_KEYPOINTS",
    "MEDIAPIPE_TO_COCO",
    "MEDIAPIPE_LANDMARKS",
    "PoseFrame",
    "PoseSequence",
    "VideoKeypoints",
    "normalize_pose",
    "extract_video_keypoints",
    "save_keypoints",
    "load_keypoints",
]
