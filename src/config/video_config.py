"""Video quality validation and metadata extraction utilities."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

# Minimum quality thresholds for pose estimation. Videos failing any threshold; will be rejected before processing to avoid wasted computation.
MIN_FPS = 24.0
MIN_WIDTH = 640
MIN_HEIGHT = 480
MIN_BITRATE_KBPS = 1000
MIN_DURATION_SECONDS = 30.0

# OpenCV codec fourcc to human-readable name mapping
FOURCC_CODECS = {
    "avc1": "h264",
    "h264": "h264",
    "hvc1": "hevc",
    "hevc": "hevc",
    "mp4v": "mpeg4",
    "xvid": "xvid",
    "mjpg": "mjpeg",
    "vp80": "vp8",
    "vp90": "vp9",
    "av01": "av1",
}


@dataclass
class VideoQualityReport:
    """Encapsulates the results of video quality validation."""

    filepath: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    bitrate_kbps: Optional[float]
    codec: str
    passes_minimum: bool
    rejection_reasons: list


def extract_video_metadata_opencv(filepath: str) -> dict:
    """
    Extract video stream metadata using OpenCV.

    This approach requires no external system dependencies like ffmpeg; making the code portable across different environments.
    """
    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        raise RuntimeError(f"Video file path couldn't be opened: {filepath}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count / fps if fps > 0 else 0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(
            [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
        ).lower().strip()
        codec = FOURCC_CODECS.get(fourcc_str, fourcc_str if fourcc_str else "unknown")

        bitrate_kbps = None
        if os.path.exists(filepath) and duration > 0:
            file_size_bytes = os.path.getsize(filepath)
            bitrate_kbps = (file_size_bytes * 8) / (duration * 1000)

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "codec": codec,
            "bitrate_kbps": bitrate_kbps,
        }
    finally:
        cap.release()


def validate_video_quality(filepath: str) -> VideoQualityReport:
    """Validate a video against minimum quality thresholds for pose estimation."""
    metadata = extract_video_metadata_opencv(filepath)

    width = metadata["width"]
    height = metadata["height"]
    fps = metadata["fps"]
    duration = metadata["duration"]
    bitrate_kbps = metadata["bitrate_kbps"]
    codec = metadata["codec"]

    rejection_reasons = []

    if fps < MIN_FPS:
        rejection_reasons.append(
            f"Frame rate {fps:.2f} fps falls below {MIN_FPS} fps minimum"
        )

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        rejection_reasons.append(
            f"Resolution {width}x{height} falls below {MIN_WIDTH}x{MIN_HEIGHT} minimum"
        )

    if bitrate_kbps is not None and bitrate_kbps < MIN_BITRATE_KBPS:
        rejection_reasons.append(
            f"Bitrate {bitrate_kbps:.0f} kbps falls below {MIN_BITRATE_KBPS} kbps minimum"
        )

    if duration < MIN_DURATION_SECONDS:
        rejection_reasons.append(
            f"Duration {duration:.1f}s falls below {MIN_DURATION_SECONDS}s minimum"
        )

    return VideoQualityReport(
        filepath = filepath,
        width = width,
        height = height,
        fps = fps,
        duration_seconds = duration,
        bitrate_kbps = bitrate_kbps,
        codec = codec,
        passes_minimum = len(rejection_reasons) == 0,
        rejection_reasons = rejection_reasons,
    )


def print_quality_report(report: VideoQualityReport) -> None:
    """Returing the video quality in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"Video Quality Report for {Path(report.filepath).name}")
    print(f"{'=' * 60}")
    print(f"Resolution:  {report.width} x {report.height}")
    print(f"Frame rate:  {report.fps:.2f} fps")
    print(f"Duration:    {report.duration_seconds:.1f} seconds")
    print(
        f"Bitrate:     {report.bitrate_kbps:.0f} kbps"
        if report.bitrate_kbps
        else "Bitrate:     unavailable"
    )
    print(f"Codec:       {report.codec}")
    print(f"{'=' * 60}")

    if report.passes_minimum:
        print("Video meets minimum quality requirements for pose estimation")
    else:
        print("Video fails minimum quality requirements:")
        for reason in report.rejection_reasons:
            print(f"  - {reason}")
    print()
