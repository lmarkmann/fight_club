"""Full pipeline orchestration for video analysis."""

from typing import Optional

import torch

from src.s00_config import validate_video_quality, print_quality_report
from src.s02_pose import extract_video_keypoints, VideoKeypoints
from src.s03_models import BoxingActionClassifier
from src.s06_visualization import sliding_window_inference, VideoAnalysis


def run_full_analysis(
    video_path: str,
    model: Optional[BoxingActionClassifier] = None,
    model_weights_path: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    window_size: int = 45,
    stride: int = 15,
    confidence_threshold: float = 0.5,
    validate_quality: bool = True,
    show_progress: bool = True,
) -> VideoAnalysis:
    """
    Run complete action classification pipeline on a video.

    Args:
        video_path: Path to input video file
        model: Pre-initialized model (optional)
        model_weights_path: Path to saved model weights (optional)
        start_time: Start analysis at this time (seconds)
        end_time: End analysis at this time (seconds)
        window_size: Frames per classification window
        stride: Step size between windows
        confidence_threshold: Minimum confidence for predictions
        validate_quality: Whether to check video quality first
        show_progress: Display progress bars

    Returns:
        VideoAnalysis containing all predictions and frame probabilities
    """
    # Step 1: Validate video quality
    if validate_quality:
        report = validate_video_quality(video_path)
        print_quality_report(report)
        if not report.passes_minimum:
            raise ValueError(
                f"Video fails quality requirements: {report.rejection_reasons}"
            )

    # Step 2: Extract pose keypoints
    keypoints: VideoKeypoints = extract_video_keypoints(
        video_path,
        start_time=start_time,
        end_time=end_time,
        show_progress=show_progress,
    )

    print(f"Extracted {len(keypoints)} frames of pose data")

    # Step 3: Initialize or load model
    if model is None:
        model = BoxingActionClassifier()
        if model_weights_path:
            model.load_state_dict(torch.load(model_weights_path, weights_only=True))
            print(f"Loaded model weights from {model_weights_path}")

    # Step 4: Run inference
    analysis: VideoAnalysis = sliding_window_inference(
        model=model,
        keypoints=keypoints,
        window_size=window_size,
        stride=stride,
        confidence_threshold=confidence_threshold,
    )

    # Summary
    print(f"\nDetected {len(analysis.predictions)} action segments:")
    for action, count in sorted(analysis.action_counts.items()):
        print(f"  {action}: {count}")

    return analysis
