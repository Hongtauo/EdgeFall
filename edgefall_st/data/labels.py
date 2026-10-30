"""Canonical OmniFall action and EdgeFall stage labels."""

from __future__ import annotations

ACTION_LABELS: dict[int, str] = {
    0: "walk",
    1: "fall",
    2: "fallen",
    3: "sit_down",
    4: "sitting",
    5: "lie_down",
    6: "lying",
    7: "stand_up",
    8: "standing",
    9: "other",
    10: "kneel_down",
    11: "kneeling",
    12: "squat_down",
    13: "squatting",
    14: "crawl",
    15: "jump",
}

STAGE_LABELS: dict[int, str] = {
    0: "normal",
    1: "falling",
    2: "fallen",
}

FALL_ACTION_ID = 1
FALLEN_ACTION_ID = 2


def action_to_stage(action_id: int) -> int:
    """Map the 16-class OmniFall taxonomy to normal/falling/fallen."""

    if action_id == FALL_ACTION_ID:
        return 1
    if action_id == FALLEN_ACTION_ID:
        return 2
    if action_id not in ACTION_LABELS:
        raise ValueError(f"Unknown OmniFall action id: {action_id}")
    return 0
