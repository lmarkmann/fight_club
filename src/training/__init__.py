"""Training module for model optimization and evaluation."""

from .trainer import EvalMetrics, train_epoch, evaluate_model
from .dataset import PoseSequenceDataset, compute_class_weights
