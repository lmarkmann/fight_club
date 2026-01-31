"""Sliding window inference for video action classification."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.s01_data.taxonomy import ACTION_LABELS
from src.s02_pose.dataclasses import VideoKeypoints
from src.s02_pose.normalization import normalize_pose


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
    frame_probs: np.ndarray  # shape: (T, num_classes) per-frame probabilities
    frame_timestamps: np.ndarray  # shape: (T,) timestamps for frame_probs
    action_counts: Dict[str, int]
    video_path: str


def sliding_window_inference(
    model: nn.Module,
    keypoints: VideoKeypoints,
    window_size: int = 45,
    stride: int = 15,
    device: Optional[torch.device] = None,
    confidence_threshold: float = 0.5,
) -> VideoAnalysis:
    """
    Run action classification across a video using overlapping sliding windows.

    Args:
        model: Trained BoxingActionClassifier
        keypoints: Extracted pose data from video
        window_size: Number of frames per classification window
        stride: Step size between consecutive windows
        device: Compute device (auto-detected if None)
        confidence_threshold: Minimum confidence to report a prediction

    Returns:
        VideoAnalysis with per-frame probabilities and detected actions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    kp_normalized = normalize_pose(keypoints.keypoints)
    T = len(kp_normalized)

    if T < window_size:
        raise ValueError(
            f"Video too short ({T} frames) for window size {window_size}"
        )

    window_starts = list(range(0, T - window_size + 1, stride))

    windows = []
    for start in window_starts:
        window = kp_normalized[start : start + window_size]
        windows.append(window)

    windows_tensor = torch.FloatTensor(np.array(windows)).to(device)

    with torch.no_grad():
        logits = model(windows_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    frame_probs = np.zeros((T, probs.shape[1]), dtype=np.float32)
    frame_counts = np.zeros(T, dtype=np.float32)

    for i, start in enumerate(window_starts):
        frame_probs[start : start + window_size] += probs[i]
        frame_counts[start : start + window_size] += 1

    frame_counts = np.maximum(frame_counts, 1)
    frame_probs /= frame_counts[:, np.newaxis]

    predictions = _extract_action_segments(
        frame_probs, keypoints.timestamps, confidence_threshold
    )

    action_counts: Dict[str, int] = {}
    for pred in predictions:
        name = pred.action_name
        action_counts[name] = action_counts.get(name, 0) + 1

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
    """
    Extract discrete action segments from per-frame probabilities.

    Uses argmax classification with segment merging: consecutive frames with
    the same predicted class are grouped, and short segments are filtered.
    """
    predictions = []

    frame_classes = frame_probs.argmax(axis=1)
    frame_confidence = frame_probs.max(axis=1)

    i = 0
    while i < len(frame_classes):
        current_class = frame_classes[i]
        start_idx = i

        while i < len(frame_classes) and frame_classes[i] == current_class:
            i += 1

        end_idx = i
        segment_len = end_idx - start_idx

        if current_class == idle_class:
            continue
        if segment_len < min_segment_frames:
            continue

        avg_confidence = frame_confidence[start_idx:end_idx].mean()
        if avg_confidence < confidence_threshold:
            continue

        predictions.append(
            ActionPrediction(
                action=int(current_class),
                action_name=ACTION_LABELS[int(current_class)],
                start_time=float(timestamps[start_idx]),
                end_time=float(timestamps[min(end_idx, len(timestamps) - 1)]),
                confidence=float(avg_confidence),
                raw_logits=frame_probs[start_idx:end_idx].mean(axis=0),
            )
        )

    return predictions
