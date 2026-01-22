"""Dataset classes for training pose-based action classifiers."""

import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseSequenceDataset(Dataset):
    """
    Dataset of labeled pose sequences.

    Each item is a pose array paired with a class label. Sequences are
    resized to a target length for batching.
    """

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
            seq = self._augment(seq)

        seq = self._resize_sequence(seq, self.target_length)
        return torch.FloatTensor(seq), label

    def _resize_sequence(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        """Resize a sequence to target length via linear interpolation."""
        current_len = seq.shape[0]
        if current_len == target_len:
            return seq

        indices = np.linspace(0, current_len - 1, target_len)
        lower = np.floor(indices).astype(int)
        upper = np.ceil(indices).astype(int)
        weight = (indices - lower)[:, np.newaxis, np.newaxis]

        return seq[lower] * (1 - weight) + seq[upper] * weight

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Apply random augmentations to increase effective dataset size."""
        # Temporal jitter: random crop within the sequence
        if random.random() < 0.5 and seq.shape[0] > 10:
            max_offset = seq.shape[0] // 10
            start = random.randint(0, max_offset)
            end = seq.shape[0] - random.randint(0, max_offset)
            seq = seq[start:end]

        # Keypoint noise: small perturbations for robustness
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, seq.shape)
            seq = seq + noise

        # Speed variation: stretch or compress time
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            new_len = int(seq.shape[0] * factor)
            seq = self._resize_sequence(seq, new_len)

        return seq
