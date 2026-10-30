"""Evaluate the V3.1 causal state machine from saved window predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from edgefall_st.runtime.v31_events import (
    V31FallStateMachine,
    V31StateMachineConfig,
    V31WindowPrediction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation-windows", type=int, default=2)
    parser.add_argument("--maximum-gap-s", type=float, default=0.6)
    parser.add_argument("--transition-window-s", type=float, default=2.0)
    parser.add_argument("--alarm-threshold", type=float, default=0.5)
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--recovery-windows", type=int, default=3)
    parser.add_argument(
        "--no-stage-transition",
        action="store_true",
        help="Disable ordered falling/onset to fallen confirmation.",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Prediction at {path}:{line_number} must be an object"
                )
            records.append(value)
    if not records:
        raise ValueError(f"No predictions found in {path}")
    return records


def _positive_paths(events: Path, evaluated_paths: set[str]) -> set[str]:
    with events.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or "path" not in rows.fieldnames:
            raise ValueError(f"Event table {events} must contain a path column")
        return {
            str(row["path"]).replace("\\", "/")
            for row in rows
            if str(row["path"]).replace("\\", "/") in evaluated_paths
        }


def _ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> dict[str, float | None]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positive_count = sum(int(label) for label in labels)
    if not scores or positive_count == 0:
        return {
            "average_precision": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "mean_precision": 0.0,
            "p90_threshold": None,
            "p95_threshold": None,
        }

    ranked = sorted(
        zip(scores, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    curve: list[dict[str, float]] = []
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(ranked):
        threshold = float(ranked[index][0])
        group_true_positives = 0
        group_size = 0
        while index < len(ranked) and ranked[index][0] == threshold:
            group_true_positives += int(ranked[index][1])
            group_size += 1
            index += 1
        true_positives += group_true_positives
        false_positives += group_size - group_true_positives
        recall = true_positives / positive_count
        precision = true_positives / max(
            true_positives + false_positives,
            1,
        )
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        curve.append(
            {
                "threshold": threshold,
                "recall": recall,
                "precision": precision,
            }
        )

    def at_recall(target: float) -> tuple[float, float | None]:
        candidates = [point for point in curve if point["recall"] >= target]
        if not candidates:
            return 0.0, None
        best = max(
            candidates,
            key=lambda point: (point["precision"], point["threshold"]),
        )
        return best["precision"], best["threshold"]

    p90, p90_threshold = at_recall(0.90)
    p95, p95_threshold = at_recall(0.95)
    return {
        "average_precision": average_precision,
        "p90": p90,
        "p95": p95,
        "mean_precision": (p90 + p95) * 0.5,
        "p90_threshold": p90_threshold,
        "p95_threshold": p95_threshold,
    }


def _classification_counts(
    scores: Sequence[float],
    labels: Sequence[bool],
    threshold: float,
) -> dict[str, int | float]:
    true_positives = sum(
        int(score >= threshold and label)
        for score, label in zip(scores, labels, strict=True)
    )
    false_positives = sum(
        int(score >= threshold and not label)
        for score, label in zip(scores, labels, strict=True)
    )
    false_negatives = sum(
        int(score < threshold and label)
        for score, label in zip(scores, labels, strict=True)
    )
    true_negatives = sum(
        int(score < threshold and not label)
        for score, label in zip(scores, labels, strict=True)
    )
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def evaluate(
    predictions_path: Path,
    events_path: Path,
    config: V31StateMachineConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl(predictions_path):
        path = str(record.get("path", "")).replace("\\", "/")
        if not path:
            raise ValueError("Every prediction must contain a non-empty path")
        grouped[path].append(record)

    paths = sorted(grouped)
    positive_paths = _positive_paths(events_path, set(paths))
    labels = [path in positive_paths for path in paths]
    raw_scores: list[float] = []
    state_scores: list[float] = []
    score_rows: list[dict[str, Any]] = []
    alarm_count = 0

    for path in paths:
        records = sorted(
            grouped[path],
            key=lambda item: (
                float(item["end_time_s"]),
                int(item.get("window_index", 0)),
            ),
        )
        machine = V31FallStateMachine(config)
        raw_score = max(float(record["fall_score"]) for record in records)
        confirmation_score = 0.0
        path_alarm_count = 0
        first_alarm_s: float | None = None
        closed_events: list[dict[str, Any]] = []
        for record in records:
            stage_values = tuple(
                float(value) for value in record["stage_probabilities"]
            )
            if len(stage_values) != 3:
                raise ValueError(
                    f"Prediction for {path} has invalid stage probabilities"
                )
            update = machine.update(
                V31WindowPrediction(
                    timestamp_s=float(record["end_time_s"]),
                    fall_score=float(record["fall_score"]),
                    stage_probabilities=stage_values,
                    onset_score=float(record["onset_score"]),
                )
            )
            confirmation_score = max(
                confirmation_score,
                update.confirmation_score,
            )
            for message in update.messages:
                if message.kind == "alarm":
                    path_alarm_count += 1
                    if first_alarm_s is None:
                        first_alarm_s = message.alarm_s
                else:
                    closed_events.append(asdict(message))
        for message in machine.flush():
            closed_events.append(asdict(message))
        alarm_count += path_alarm_count
        raw_scores.append(raw_score)
        state_scores.append(confirmation_score)
        score_rows.append(
            {
                "path": path,
                "target": path in positive_paths,
                "raw_max_score": raw_score,
                "v31_score": confirmation_score,
                "alarms_at_fixed_threshold": path_alarm_count,
                "first_alarm_s": first_alarm_s,
                "events_at_fixed_threshold": closed_events,
            }
        )

    threshold = config.alarm_threshold
    raw_positive = {
        path for path, score in zip(paths, raw_scores, strict=True)
        if score >= threshold
    }
    state_positive = {
        path for path, score in zip(paths, state_scores, strict=True)
        if score >= threshold
    }
    result = {
        "prediction_count": sum(len(records) for records in grouped.values()),
        "evaluated_videos": len(paths),
        "positive_videos": len(positive_paths),
        "configuration": asdict(config),
        "score_definition": (
            "max over causal windows of max(rolling-min fall score, "
            "ordered falling/onset-to-fallen score)"
        ),
        "raw_max_baseline": {
            **_ranking_metrics(raw_scores, labels),
            "at_fixed_threshold": _classification_counts(
                raw_scores,
                labels,
                threshold,
            ),
        },
        "v31_state_machine": {
            **_ranking_metrics(state_scores, labels),
            "at_fixed_threshold": _classification_counts(
                state_scores,
                labels,
                threshold,
            ),
            "alarms_at_fixed_threshold": alarm_count,
        },
        "fixed_threshold_effect": {
            "suppressed_negative_videos": len(
                (raw_positive - state_positive) - positive_paths
            ),
            "suppressed_positive_videos": len(
                (raw_positive - state_positive) & positive_paths
            ),
            "raw_positive_videos": len(raw_positive),
            "v31_positive_videos": len(state_positive),
        },
    }
    return result, score_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = V31StateMachineConfig(
        confirmation_windows=args.confirmation_windows,
        maximum_gap_s=args.maximum_gap_s,
        transition_window_s=args.transition_window_s,
        alarm_threshold=args.alarm_threshold,
        release_threshold=args.release_threshold,
        recovery_windows=args.recovery_windows,
        use_stage_transition=not args.no_stage_transition,
    )
    metrics, score_rows = evaluate(args.predictions, args.events, config)
    result = {
        "predictions": str(args.predictions.resolve()),
        "events": str(args.events.resolve()),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.output)

    scores_output = args.output.with_name(
        f"{args.output.stem}_scores.jsonl"
    )
    scores_temporary = scores_output.with_suffix(scores_output.suffix + ".tmp")
    with scores_temporary.open("w", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores_temporary.replace(scores_output)
    result["output"] = str(args.output.resolve())
    result["scores_output"] = str(scores_output.resolve())
    return result


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
