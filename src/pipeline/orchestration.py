"""Full video analysis pipeline."""

from typing import Optional

import torch

from src.config import validate_video_quality, print_quality_report
from src.pose import extract_video_keypoints, VideoKeypoints
from src.models import BoxingActionClassifier
from src.visualization import sliding_window_inference, VideoAnalysis


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
    """Run complete action classification pipeline on a video."""
    if validate_quality:
        report = validate_video_quality(video_path)
        print_quality_report(report)
        if not report.passes_minimum:
            raise ValueError(f"Video fails quality requirements: {report.rejection_reasons}")

    keypoints: VideoKeypoints = extract_video_keypoints(
        video_path, start_time=start_time, end_time=end_time, show_progress=show_progress,
    )
    print(f"Extracted {len(keypoints)} frames of pose data")

    if model is None:
        model = BoxingActionClassifier()
        if model_weights_path:
            model.load_state_dict(torch.load(model_weights_path, weights_only=True))

    analysis: VideoAnalysis = sliding_window_inference(
        model=model, keypoints=keypoints, window_size=window_size,
        stride=stride, confidence_threshold=confidence_threshold,
    )

    print(f"\nDetected {len(analysis.predictions)} action segments:")
    for action, count in sorted(analysis.action_counts.items()):
        print(f"  {action}: {count}")

    return analysis
