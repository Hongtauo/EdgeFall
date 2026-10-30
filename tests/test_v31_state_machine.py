from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgefall_st.evaluation.v31_state_machine import evaluate
from edgefall_st.runtime.v31_events import (
    V31EventState,
    V31FallStateMachine,
    V31StateMachineConfig,
    V31WindowPrediction,
    v31_sequence_score,
)


def prediction(
    timestamp_s: float,
    fall_score: float,
    *,
    stage: tuple[float, float, float] = (0.9, 0.05, 0.05),
    onset: float = 0.05,
) -> V31WindowPrediction:
    return V31WindowPrediction(
        timestamp_s=timestamp_s,
        fall_score=fall_score,
        stage_probabilities=stage,
        onset_score=onset,
    )


def test_isolated_high_score_does_not_alarm() -> None:
    machine = V31FallStateMachine()
    updates = [
        machine.update(prediction(0.0, 0.1)),
        machine.update(
            prediction(0.4, 0.9, stage=(0.05, 0.9, 0.05), onset=0.8)
        ),
        machine.update(prediction(0.8, 0.1)),
    ]

    assert not [message for update in updates for message in update.messages]
    assert max(update.confirmation_score for update in updates) < 0.5


def test_persistent_fall_emits_one_alarm_and_one_close() -> None:
    machine = V31FallStateMachine(
        V31StateMachineConfig(recovery_windows=2)
    )
    messages = []
    messages.extend(
        machine.update(
            prediction(0.0, 0.8, stage=(0.05, 0.85, 0.1), onset=0.8)
        ).messages
    )
    confirmed = machine.update(
        prediction(0.4, 0.75, stage=(0.05, 0.15, 0.8), onset=0.2)
    )
    messages.extend(confirmed.messages)
    messages.extend(machine.update(prediction(0.8, 0.05)).messages)
    messages.extend(machine.update(prediction(1.2, 0.05)).messages)

    assert confirmed.state is V31EventState.ALARMED
    assert confirmed.confirmation_score == pytest.approx(0.75)
    assert [message.kind for message in messages] == ["alarm", "closed"]
    assert messages[0].alarm_s == pytest.approx(0.4)


def test_ordered_stage_transition_can_bridge_one_low_window() -> None:
    config = V31StateMachineConfig(
        confirmation_windows=3,
        transition_window_s=1.5,
    )
    predictions = [
        prediction(0.0, 0.8, stage=(0.05, 0.9, 0.05), onset=0.8),
        prediction(0.4, 0.1),
        prediction(0.8, 0.85, stage=(0.05, 0.1, 0.85)),
    ]

    with_transition = v31_sequence_score(predictions, config)
    without_transition = v31_sequence_score(
        predictions,
        V31StateMachineConfig(
            confirmation_windows=3,
            transition_window_s=1.5,
            use_stage_transition=False,
        ),
    )

    assert with_transition == pytest.approx(0.8)
    assert without_transition == pytest.approx(0.1)


def test_predictions_must_be_chronological() -> None:
    machine = V31FallStateMachine()
    machine.update(prediction(1.0, 0.2))

    with pytest.raises(ValueError, match="non-decreasing"):
        machine.update(prediction(0.5, 0.2))


def test_flush_clears_evidence_before_reuse() -> None:
    machine = V31FallStateMachine()
    machine.update(prediction(1.0, 0.9))
    machine.flush()

    update = machine.update(prediction(0.0, 0.9))

    assert update.confirmation_score == 0.0
    assert not update.messages


def test_evaluator_suppresses_single_peak_negative(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    records = [
        {
            "path": "fall/positive",
            "window_index": 0,
            "end_time_s": 0.0,
            "fall_score": 0.8,
            "stage_probabilities": [0.05, 0.9, 0.05],
            "onset_score": 0.8,
        },
        {
            "path": "fall/positive",
            "window_index": 1,
            "end_time_s": 0.4,
            "fall_score": 0.8,
            "stage_probabilities": [0.05, 0.1, 0.85],
            "onset_score": 0.1,
        },
        {
            "path": "other/negative",
            "window_index": 0,
            "end_time_s": 0.0,
            "fall_score": 0.95,
            "stage_probabilities": [0.05, 0.9, 0.05],
            "onset_score": 0.9,
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    events_path = tmp_path / "events.csv"
    events_path.write_text(
        "path,start_s,end_s\nfall/positive,0.0,1.0\n",
        encoding="utf-8",
    )

    result, score_rows = evaluate(
        predictions_path,
        events_path,
        V31StateMachineConfig(),
    )

    assert result["raw_max_baseline"]["average_precision"] == pytest.approx(0.5)
    assert result["v31_state_machine"]["average_precision"] == pytest.approx(1.0)
    assert result["fixed_threshold_effect"]["suppressed_negative_videos"] == 1
    assert result["fixed_threshold_effect"]["suppressed_positive_videos"] == 0
    assert {row["path"] for row in score_rows} == {
        "fall/positive",
        "other/negative",
    }
    positive = next(row for row in score_rows if row["target"])
    assert len(positive["events_at_fixed_threshold"]) == 1
    assert positive["events_at_fixed_threshold"][0]["kind"] == "closed"
