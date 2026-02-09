"""Training and evaluation utilities with per-class metrics."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.taxonomy import ACTION_LABELS


@dataclass
class EvalMetrics:
    """Evaluation results with per-class breakdown."""
    loss: float
    accuracy: float
    per_class: Dict[str, Dict[str, float]]  # {class_name: {precision, recall, f1}}
    confusion_matrix: np.ndarray             # (num_classes, num_classes)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
    scheduler: object = None,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        correct += (logits.argmax(dim=-1) == batch_y).sum().item()
        total += batch_y.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 13,
) -> EvalMetrics:
    """Evaluate model with per-class precision, recall, F1 and confusion matrix."""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        total_loss += criterion(logits, batch_y).item() * batch_x.size(0)
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(batch_y.cpu())
        total += batch_y.size(0)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # Confusion matrix: rows=true, cols=predicted
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    per_class = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[ACTION_LABELS[i]] = {"precision": prec, "recall": rec, "f1": f1}

    accuracy = (preds == labels).mean() if total > 0 else 0.0

    return EvalMetrics(
        loss=total_loss / total,
        accuracy=float(accuracy),
        per_class=per_class,
        confusion_matrix=cm,
    )
