"""Dataset classes for training pose-based action classifiers."""

import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.taxonomy import LEAD_REAR_SWAP
from src.pose.constants import COCO_KEYPOINTS

# Left↔right keypoint index pairs for horizontal flip
_LR_PAIRS = [
    (COCO_KEYPOINTS["left_eye"], COCO_KEYPOINTS["right_eye"]),
    (COCO_KEYPOINTS["left_ear"], COCO_KEYPOINTS["right_ear"]),
    (COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["right_shoulder"]),
    (COCO_KEYPOINTS["left_elbow"], COCO_KEYPOINTS["right_elbow"]),
    (COCO_KEYPOINTS["left_wrist"], COCO_KEYPOINTS["right_wrist"]),
    (COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["right_hip"]),
    (COCO_KEYPOINTS["left_knee"], COCO_KEYPOINTS["right_knee"]),
    (COCO_KEYPOINTS["left_ankle"], COCO_KEYPOINTS["right_ankle"]),
]


def _resize_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Resize a sequence to target length via linear interpolation."""
    t = seq.shape[0]
    if t == target_len:
        return seq
    indices = np.linspace(0, t - 1, target_len)
    lo, hi = np.floor(indices).astype(int), np.ceil(indices).astype(int)
    w = (indices - lo)[:, None, None]
    return seq[lo] * (1 - w) + seq[hi] * w


class PoseSequenceDataset(Dataset):
    """Dataset of labeled pose sequences with augmentation support."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        target_length: int = 45,
        augment: bool = False,
    ):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels
        self.target_length = target_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = self.sequences[idx].copy()
        label = self.labels[idx]

        if self.augment:
            seq, label = self._augment(seq, label)

        seq = _resize_sequence(seq, self.target_length)
        return torch.FloatTensor(seq), label

    def _augment(self, seq: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
        """Apply random augmentations."""
        # Temporal jitter
        if random.random() < 0.5 and seq.shape[0] > 10:
            offset = seq.shape[0] // 10
            start = random.randint(0, offset)
            seq = seq[start : seq.shape[0] - random.randint(0, offset)]

        # Keypoint noise
        if random.random() < 0.5:
            seq = seq + np.random.normal(0, 0.02, seq.shape)

        # Speed variation
        if random.random() < 0.3:
            seq = _resize_sequence(seq, int(seq.shape[0] * random.uniform(0.8, 1.2)))

        # Horizontal flip with lead↔rear label swap
        if random.random() < 0.5:
            seq = seq.copy()
            seq[..., 0] *= -1  # negate x (already hip-centered)
            for l, r in _LR_PAIRS:
                seq[:, [l, r]] = seq[:, [r, l]]
            label = LEAD_REAR_SWAP[label]

        # Joint dropout: zero out 1-2 random keypoints to simulate occlusion
        if random.random() < 0.3:
            drop_idx = np.random.choice(17, size=random.randint(1, 2), replace=False)
            seq[:, drop_idx] = 0.0

        return seq, label


def compute_class_weights(labels: List[int], num_classes: int = 13) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.FloatTensor(weights)
