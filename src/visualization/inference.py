"""Sliding window inference for video action classification."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import median_filter

from src.data.taxonomy import ACTION_LABELS
from src.pose.dataclasses import VideoKeypoints
from src.pose.normalization import normalize_pose, compute_motion_features


@dataclass
class ActionPrediction:
    """A single action prediction with temporal bounds and confidence."""
    action: int
    action_name: str
    start_time: float
    end_time: float
    confidence: float
    raw_logits: np.ndarray


@dataclass
class VideoAnalysis:
    """Complete action analysis results for a video."""
    predictions: List[ActionPrediction]
    frame_probs: np.ndarray     # (T, num_classes)
    frame_timestamps: np.ndarray  # (T,)
    action_counts: Dict[str, int]
    video_path: str


def sliding_window_inference(
    model: nn.Module,
    keypoints: VideoKeypoints,
    window_size: int = 45,
    stride: int = 15,
    device: Optional[torch.device] = None,
    confidence_threshold: float = 0.5,
    temporal_smooth_window: int = 7,
) -> VideoAnalysis:
    """Run action classification across a video using overlapping sliding windows."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Normalize and compute motion features (pos + vel + acc)
    kp = compute_motion_features(normalize_pose(keypoints.keypoints))
    T = len(kp)

    if T < window_size:
        raise ValueError(f"Video too short ({T} frames) for window size {window_size}")

    window_starts = list(range(0, T - window_size + 1, stride))
    windows = np.array([kp[s : s + window_size] for s in window_starts])

    with torch.no_grad():
        probs = F.softmax(model(torch.FloatTensor(windows).to(device)), dim=-1).cpu().numpy()

    # Average overlapping window predictions
    frame_probs = np.zeros((T, probs.shape[1]), dtype=np.float32)
    frame_counts = np.zeros(T, dtype=np.float32)
    for i, s in enumerate(window_starts):
        frame_probs[s : s + window_size] += probs[i]
        frame_counts[s : s + window_size] += 1
    frame_probs /= np.maximum(frame_counts, 1)[:, None]

    # Temporal smoothing: median filter on per-class probabilities
    if temporal_smooth_window > 1:
        frame_probs = median_filter(frame_probs, size=(temporal_smooth_window, 1))

    predictions = _extract_action_segments(
        frame_probs, keypoints.timestamps, confidence_threshold
    )

    action_counts: Dict[str, int] = {}
    for pred in predictions:
        action_counts[pred.action_name] = action_counts.get(pred.action_name, 0) + 1

    return VideoAnalysis(
        predictions=predictions,
        frame_probs=frame_probs,
        frame_timestamps=keypoints.timestamps,
        action_counts=action_counts,
        video_path=keypoints.video_path,
    )


def _extract_action_segments(
    frame_probs: np.ndarray,
    timestamps: np.ndarray,
    confidence_threshold: float,
    min_segment_frames: int = 5,
    idle_class: int = 12,
) -> List[ActionPrediction]:
    """Extract discrete action segments from per-frame probabilities."""
    predictions = []
    classes = frame_probs.argmax(axis=1)
    confidences = frame_probs.max(axis=1)

    i = 0
    while i < len(classes):
        cls, start = classes[i], i
        while i < len(classes) and classes[i] == cls:
            i += 1

        if cls == idle_class or (i - start) < min_segment_frames:
            continue

        avg_conf = confidences[start:i].mean()
        if avg_conf < confidence_threshold:
            continue

        predictions.append(ActionPrediction(
            action=int(cls),
            action_name=ACTION_LABELS[int(cls)],
            start_time=float(timestamps[start]),
            end_time=float(timestamps[min(i, len(timestamps) - 1)]),
            confidence=float(avg_conf),
            raw_logits=frame_probs[start:i].mean(axis=0),
        ))

    return predictions
