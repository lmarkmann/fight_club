"""Fight Club: Pose-based action classification for combat sports."""

from src.01_config import (
    VideoQualityReport,
    validate_video_quality,
    print_quality_report,
)
from src.02_data import BoxingAction, ACTION_LABELS
from src.03_pose import (
    VideoKeypoints,
    extract_video_keypoints,
    normalize_pose,
)
from src.04_models import BoxingActionClassifier
from src.05_training import train_epoch, evaluate_model, PoseSequenceDataset
from src.06_visualization import (
    ActionPrediction,
    VideoAnalysis,
    sliding_window_inference,
)
from src.07_pipeline import run_full_analysis

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
