"""Training module for model optimization and evaluation."""

from .trainer import train_epoch, evaluate_model
from .dataset import PoseSequenceDataset

__all__ = [
    "train_epoch",
    "evaluate_model",
    "PoseSequenceDataset",
]
