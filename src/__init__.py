"""Fight Club: Pose-based action classification for combat sports."""

from src.config import (
    VideoQualityReport,
    validate_video_quality,
    print_quality_report,
)
from src.data import BoxingAction, ACTION_LABELS
from src.pose import (
    VideoKeypoints,
    extract_video_keypoints,
    normalize_pose,
)
from src.models import BoxingActionClassifier
from src.training import train_epoch, evaluate_model, PoseSequenceDataset
from src.visualization import (
    ActionPrediction,
    VideoAnalysis,
    sliding_window_inference,
)
from src.pipeline import run_full_analysis

__all__ = [
    # Config
    "VideoQualityReport",
    "validate_video_quality",
    "print_quality_report",
    # Data
    "BoxingAction",
    "ACTION_LABELS",
    # Pose
    "VideoKeypoints",
    "extract_video_keypoints",
    "normalize_pose",
    # Models
    "BoxingActionClassifier",
    # Training
    "train_epoch",
    "evaluate_model",
    "PoseSequenceDataset",
    # Visualization
    "ActionPrediction",
    "VideoAnalysis",
    "sliding_window_inference",
    # Pipeline
    "run_full_analysis",
]
