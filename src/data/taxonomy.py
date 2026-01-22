"""Boxing action taxonomy and label definitions."""

from enum import IntEnum
from typing import Dict, List


class BoxingAction(IntEnum):
    """
    Thirteen-class taxonomy for boxing actions.

    The naming convention follows HAND_TRAJECTORY_TARGET where applicable.
    Uppercuts do not distinguish head from body targets because their rising
    trajectory makes the distinction depend more on distance than intent.
    """

    JAB_HEAD = 0
    JAB_BODY = 1
    CROSS_HEAD = 2
    CROSS_BODY = 3
    LEAD_HOOK_HEAD = 4
    LEAD_HOOK_BODY = 5
    REAR_HOOK_HEAD = 6
    REAR_HOOK_BODY = 7
    LEAD_UPPERCUT = 8
    REAR_UPPERCUT = 9
    OVERHAND = 10
    DEFENSIVE = 11
    IDLE = 12


ACTION_LABELS: Dict[int, str] = {
    0: "Jab (Head)",
    1: "Jab (Body)",
    2: "Cross (Head)",
    3: "Cross (Body)",
    4: "Lead Hook (Head)",
    5: "Lead Hook (Body)",
    6: "Rear Hook (Head)",
    7: "Rear Hook (Body)",
    8: "Lead Uppercut",
    9: "Rear Uppercut",
    10: "Overhand",
    11: "Defensive Movement",
    12: "Idle/Stance",
}


# Keyboard shortcuts for the labeling interface
LABEL_HOTKEYS: Dict[str, int] = {
    "1": BoxingAction.JAB_HEAD,
    "2": BoxingAction.JAB_BODY,
    "3": BoxingAction.CROSS_HEAD,
    "4": BoxingAction.CROSS_BODY,
    "5": BoxingAction.LEAD_HOOK_HEAD,
    "6": BoxingAction.LEAD_HOOK_BODY,
    "7": BoxingAction.REAR_HOOK_HEAD,
    "8": BoxingAction.REAR_HOOK_BODY,
    "9": BoxingAction.LEAD_UPPERCUT,
    "0": BoxingAction.REAR_UPPERCUT,
    "o": BoxingAction.OVERHAND,
    "d": BoxingAction.DEFENSIVE,
    "i": BoxingAction.IDLE,
}


def get_action_hierarchy() -> Dict[str, List[BoxingAction]]:
    """Group actions by trajectory type for hierarchical analysis."""
    return {
        "straight": [
            BoxingAction.JAB_HEAD,
            BoxingAction.JAB_BODY,
            BoxingAction.CROSS_HEAD,
            BoxingAction.CROSS_BODY,
        ],
        "hook": [
            BoxingAction.LEAD_HOOK_HEAD,
            BoxingAction.LEAD_HOOK_BODY,
            BoxingAction.REAR_HOOK_HEAD,
            BoxingAction.REAR_HOOK_BODY,
        ],
        "uppercut": [
            BoxingAction.LEAD_UPPERCUT,
            BoxingAction.REAR_UPPERCUT,
        ],
        "other": [
            BoxingAction.OVERHAND,
            BoxingAction.DEFENSIVE,
            BoxingAction.IDLE,
        ],
    }
