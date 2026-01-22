"""Video pose extraction using MediaPipe."""

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm.auto import tqdm

from .constants import MEDIAPIPE_TO_COCO
from .dataclasses import VideoKeypoints


def extract_video_keypoints(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    subsample: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    show_progress: bool = True,
) -> VideoKeypoints:
    """
    Extract pose keypoints from every frame of a video.

    Args:
        video_path: Path to video file
        start_time: Start extraction at this time (seconds). None = beginning.
        end_time: Stop extraction at this time (seconds). None = end.
        subsample: Process every Nth frame (1 = all frames, 2 = half, etc.)
        min_detection_confidence: MediaPipe detection threshold
        min_tracking_confidence: MediaPipe tracking threshold
        show_progress: Display progress bar

    Returns:
        VideoKeypoints containing pose sequences for the entire video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoints_list = []
    confidence_list = []
    timestamps_list = []

    frame_idx = start_frame
    frames_to_process = range(start_frame, end_frame, subsample)

    iterator = (
        tqdm(frames_to_process, desc="Extracting poses")
        if show_progress
        else frames_to_process
    )

    try:
        for target_frame in iterator:
            while frame_idx < target_frame:
                cap.grab()
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            kp = np.zeros((17, 2), dtype=np.float32)
            conf = np.zeros(17, dtype=np.float32)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
                    if mp_idx < len(landmarks):
                        lm = landmarks[mp_idx]
                        kp[coco_idx] = [lm.x, lm.y]
                        conf[coco_idx] = lm.visibility

            keypoints_list.append(kp)
            confidence_list.append(conf)
            timestamps_list.append(target_frame / fps)

    finally:
        cap.release()
        pose.close()

    effective_fps = fps / subsample

    return VideoKeypoints(
        keypoints=np.array(keypoints_list),
        confidence=np.array(confidence_list),
        timestamps=np.array(timestamps_list),
        fps=effective_fps,
        video_path=video_path,
    )


def save_keypoints(vk: VideoKeypoints, path: str) -> None:
    """Save extracted keypoints to a compressed numpy file."""
    np.savez_compressed(
        path,
        keypoints=vk.keypoints,
        confidence=vk.confidence,
        timestamps=vk.timestamps,
        fps=vk.fps,
        video_path=vk.video_path,
    )
    print(f"Saved keypoints to {path}")


def load_keypoints(path: str) -> VideoKeypoints:
    """Load keypoints from a saved numpy file."""
    data = np.load(path, allow_pickle=True)
    return VideoKeypoints(
        keypoints=data["keypoints"],
        confidence=data["confidence"],
        timestamps=data["timestamps"],
        fps=float(data["fps"]),
        video_path=str(data["video_path"]),
    )
