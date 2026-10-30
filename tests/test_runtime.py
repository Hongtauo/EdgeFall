from __future__ import annotations

import pytest
import torch

from edgefall_st.runtime.association import bbox_iou, greedy_match
from edgefall_st.runtime.events import EventConfig, FallEventManager
from edgefall_st.runtime.fygr_online import _attention_masses
from edgefall_st.runtime.tracker import MultiObjectTracker, TrackerConfig
from edgefall_st.runtime.types import Detection


def detection(
    bbox: tuple[float, float, float, float],
    *,
    stage: tuple[float, float, float] = (0.9, 0.05, 0.05),
    onset: float = 0.05,
    embedding: tuple[float, float] = (1.0, 0.0),
) -> Detection:
    return Detection(
        bbox=bbox,
        person_score=0.9,
        stage_probabilities=stage,
        onset_score=onset,
        embedding=embedding,
    )


def test_bbox_iou() -> None:
    assert bbox_iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == 1.0
    assert bbox_iou((0.0, 0.0, 0.2, 0.2), (0.8, 0.8, 1.0, 1.0)) == 0.0


def test_tracker_keeps_identity_across_frames() -> None:
    tracker = MultiObjectTracker(TrackerConfig(minimum_hits=1))
    first, _ = tracker.update([detection((0.1, 0.1, 0.3, 0.5))])
    second, _ = tracker.update([detection((0.11, 0.1, 0.31, 0.5))])

    assert first[0].track_id == second[0].track_id


def test_state_machine_emits_one_alarm_and_one_close() -> None:
    tracker = MultiObjectTracker(TrackerConfig(minimum_hits=1))
    manager = FallEventManager(
        EventConfig(confirmation_windows=2, recovery_windows=2)
    )
    messages = []
    for index in range(3):
        tracks, expired = tracker.update(
            [
                detection(
                    (0.1, 0.1, 0.3, 0.5),
                    stage=(0.05, 0.8, 0.15),
                    onset=0.7,
                )
            ]
        )
        messages.extend(manager.update(index * 0.1, tracks, expired))
    for index in range(2):
        tracks, expired = tracker.update(
            [detection((0.1, 0.1, 0.3, 0.5))]
        )
        messages.extend(manager.update(0.3 + index * 0.1, tracks, expired))

    assert [message.kind for message in messages] == ["alarm", "closed"]


def test_online_attention_masses_support_summary_attention() -> None:
    global_mass, roi_mass = _attention_masses(
        torch.tensor([[0.3, 0.7]]),
        use_global=True,
        use_roi=True,
        visual_slots=8,
    )

    assert global_mass == pytest.approx(0.3)
    assert roi_mass == pytest.approx(0.7)


def test_online_attention_masses_support_temporal_attention() -> None:
    attention = torch.tensor(
        [
            [
                [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.1, 0.1, 0.2, 0.15, 0.2, 0.25],
            ]
        ]
    )
    global_mass, roi_mass = _attention_masses(
        attention,
        use_global=True,
        use_roi=True,
        visual_slots=3,
    )

    assert global_mass == pytest.approx(0.4)
    assert roi_mass == pytest.approx(0.6)
