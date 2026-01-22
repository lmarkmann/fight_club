"""Configuration module for video quality validation and processing constants."""

from .video_config import (
    VideoQualityReport,
    MIN_FPS,
    MIN_WIDTH,
    MIN_HEIGHT,
    MIN_BITRATE_KBPS,
    MIN_DURATION_SECONDS,
    FOURCC_CODECS,
    extract_video_metadata_opencv,
    validate_video_quality,
    print_quality_report,
)

__all__ = [
    "VideoQualityReport",
    "MIN_FPS",
    "MIN_WIDTH",
    "MIN_HEIGHT",
    "MIN_BITRATE_KBPS",
    "MIN_DURATION_SECONDS",
    "FOURCC_CODECS",
    "extract_video_metadata_opencv",
    "validate_video_quality",
    "print_quality_report",
]
