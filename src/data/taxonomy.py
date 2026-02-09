"""Boxing action taxonomy and label definitions."""

from enum import IntEnum
from typing import Dict, List


class BoxingAction(IntEnum):
    """Thirteen-class taxonomy: HAND_TRAJECTORY_TARGET convention."""

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


ACTION_LABELS: Dict[int, str] = {a.value: a.name.replace("_", " ").title() for a in BoxingAction}

LABEL_HOTKEYS: Dict[str, int] = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
    "7": 6, "8": 7, "9": 8, "0": 9, "o": 10, "d": 11, "i": 12,
}

# Lead↔rear label swaps for horizontal flip augmentation.
# Maps each lead-hand action to its rear-hand equivalent and vice versa.
# Non-handed actions (overhand, defensive, idle) map to themselves.
LEAD_REAR_SWAP: Dict[int, int] = {
    0: 2, 2: 0,    # jab_head ↔ cross_head
    1: 3, 3: 1,    # jab_body ↔ cross_body
    4: 6, 6: 4,    # lead_hook_head ↔ rear_hook_head
    5: 7, 7: 5,    # lead_hook_body ↔ rear_hook_body
    8: 9, 9: 8,    # lead_uppercut ↔ rear_uppercut
    10: 10, 11: 11, 12: 12,
}


def get_action_hierarchy() -> Dict[str, List[BoxingAction]]:
    """Group actions by trajectory type."""
    return {
        "straight": [BoxingAction.JAB_HEAD, BoxingAction.JAB_BODY,
                      BoxingAction.CROSS_HEAD, BoxingAction.CROSS_BODY],
        "hook": [BoxingAction.LEAD_HOOK_HEAD, BoxingAction.LEAD_HOOK_BODY,
                 BoxingAction.REAR_HOOK_HEAD, BoxingAction.REAR_HOOK_BODY],
        "uppercut": [BoxingAction.LEAD_UPPERCUT, BoxingAction.REAR_UPPERCUT],
        "other": [BoxingAction.OVERHAND, BoxingAction.DEFENSIVE, BoxingAction.IDLE],
    }
