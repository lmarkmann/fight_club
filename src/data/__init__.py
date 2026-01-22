"""Data module containing action taxonomy, dataset classes, and data preparation."""

from .taxonomy import (
    BoxingAction,
    ACTION_LABELS,
    LABEL_HOTKEYS,
    get_action_hierarchy,
)

__all__ = [
    "BoxingAction",
    "ACTION_LABELS",
    "LABEL_HOTKEYS",
    "get_action_hierarchy",
]
