"""Visualization module for pose and classification results."""

from .inference import (
    ActionPrediction,
    VideoAnalysis,
    sliding_window_inference,
)

__all__ = [
    "ActionPrediction",
    "VideoAnalysis",
    "sliding_window_inference",
]
