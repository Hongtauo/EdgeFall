from __future__ import annotations

from edgefall_st.evaluation.events import Event, evaluate_p90_p95, match_events


def test_duplicate_prediction_counts_as_false_positive() -> None:
    ground_truth = [Event("video", 1.0, 2.0)]
    predictions = [
        Event("video", 1.0, 2.0, 0.9),
        Event("video", 1.1, 2.0, 0.8),
    ]

    metrics = match_events(ground_truth, predictions)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0


def test_p90_p95_are_computed_from_event_scores() -> None:
    ground_truth = [
        Event("a", 1.0, 2.0),
        Event("b", 1.0, 2.0),
    ]
    predictions = [
        Event("a", 1.0, 2.0, 0.9),
        Event("b", 1.0, 2.0, 0.8),
        Event("c", 1.0, 2.0, 0.1),
    ]

    result = evaluate_p90_p95(ground_truth, predictions)

    assert result.p90 == 1.0
    assert result.p95 == 1.0
    assert result.mean_precision == 1.0
