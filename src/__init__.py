"""Fight Club: Pose-based action classification for combat sports."""

from src.config import VideoQualityReport, validate_video_quality, print_quality_report
from src.data import BoxingAction, ACTION_LABELS, LEAD_REAR_SWAP
from src.pose import VideoKeypoints, extract_video_keypoints, normalize_pose, compute_motion_features
from src.models import BoxingActionClassifier, TemporalAttentionPool
from src.training import EvalMetrics, train_epoch, evaluate_model, PoseSequenceDataset, compute_class_weights
from src.visualization import ActionPrediction, VideoAnalysis, sliding_window_inference
from src.pipeline import run_full_analysis
