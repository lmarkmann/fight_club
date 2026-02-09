"""Pose extraction module for MediaPipe-based keypoint detection."""

from .constants import COCO_KEYPOINTS, BOXING_KEYPOINTS, MEDIAPIPE_TO_COCO
from .dataclasses import VideoKeypoints
from .normalization import normalize_pose, compute_motion_features
from .extraction import extract_video_keypoints, save_keypoints, load_keypoints
