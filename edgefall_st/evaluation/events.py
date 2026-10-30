"""Configurable event matching and high-recall precision metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Event:
    video_id: str
    start_s: float
    end_s: float
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.start_s > self.end_s:
            raise ValueError("Event start must not be after event end")


@dataclass(frozen=True)
class EventMetrics:
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float


@dataclass(frozen=True)
class P90P95Result:
    p90: float
    p95: float
    mean_precision: float
    p90_threshold: float | None
    p95_threshold: float | None
    curve: tuple[EventMetrics, ...]


def temporal_iou(first: Event, second: Event) -> float:
    intersection = max(
        0.0,
        min(first.end_s, second.end_s) - max(first.start_s, second.start_s),
    )
    union = max(first.end_s, second.end_s) - min(
        first.start_s,
        second.start_s,
    )
    if union == 0:
        return 1.0 if first.start_s == second.start_s else 0.0
    return intersection / union


def match_events(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    *,
    score_threshold: float = 0.0,
    minimum_temporal_iou: float = 0.1,
    onset_tolerance_s: float | None = None,
) -> EventMetrics:
    """Greedily match one prediction to at most one ground-truth event."""

    if not 0 <= minimum_temporal_iou <= 1:
        raise ValueError("minimum_temporal_iou must be in [0, 1]")
    selected = [
        prediction
        for prediction in predictions
        if prediction.score >= score_threshold
    ]
    selected.sort(key=lambda event: (-event.score, event.video_id, event.start_s))

    ground_truth_by_video: dict[str, list[tuple[int, Event]]] = {}
    for index, event in enumerate(ground_truth):
        ground_truth_by_video.setdefault(event.video_id, []).append((index, event))

    matched_ground_truth: set[int] = set()
    true_positives = 0
    false_positives = 0
    for prediction in selected:
        candidates: list[tuple[float, int]] = []
        for index, target in ground_truth_by_video.get(prediction.video_id, []):
            if index in matched_ground_truth:
                continue
            overlap = temporal_iou(target, prediction)
            onset_match = (
                onset_tolerance_s is not None
                and abs(prediction.start_s - target.start_s) <= onset_tolerance_s
            )
            if overlap >= minimum_temporal_iou or onset_match:
                candidates.append((overlap, index))
        if candidates:
            _, selected_index = max(candidates)
            matched_ground_truth.add(selected_index)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    return EventMetrics(
        threshold=score_threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
    )


def precision_recall_curve(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    *,
    minimum_temporal_iou: float = 0.1,
    onset_tolerance_s: float | None = None,
) -> tuple[EventMetrics, ...]:
    thresholds = sorted({prediction.score for prediction in predictions}, reverse=True)
    if not thresholds:
        thresholds = [inf]
    return tuple(
        match_events(
            ground_truth,
            predictions,
            score_threshold=threshold,
            minimum_temporal_iou=minimum_temporal_iou,
            onset_tolerance_s=onset_tolerance_s,
        )
        for threshold in thresholds
    )


def _precision_at_recall(
    curve: Iterable[EventMetrics],
    target_recall: float,
) -> tuple[float, float | None]:
    candidates = [point for point in curve if point.recall >= target_recall]
    if not candidates:
        return 0.0, None
    best = max(candidates, key=lambda point: (point.precision, point.threshold))
    return best.precision, best.threshold


def evaluate_p90_p95(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    *,
    minimum_temporal_iou: float = 0.1,
    onset_tolerance_s: float | None = None,
) -> P90P95Result:
    """Compute a provisional P90/P95 until the official matcher is available."""

    curve = precision_recall_curve(
        ground_truth,
        predictions,
        minimum_temporal_iou=minimum_temporal_iou,
        onset_tolerance_s=onset_tolerance_s,
    )
    p90, p90_threshold = _precision_at_recall(curve, 0.90)
    p95, p95_threshold = _precision_at_recall(curve, 0.95)
    return P90P95Result(
        p90=p90,
        p95=p95,
        mean_precision=(p90 + p95) / 2.0,
        p90_threshold=p90_threshold,
        p95_threshold=p95_threshold,
        curve=curve,
    )
