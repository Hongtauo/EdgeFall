"""Analyze frozen external-test predictions without changing the protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from external_eval.fall_video_dataset.common import (
    binary_ranking_metrics,
    fixed_threshold_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-errors", type=int, default=25)
    return parser


def _source_family(video_path: str) -> str:
    stem = Path(video_path).stem
    if re.match(r"^\d{8}_?\d{6}$", stem):
        return "timestamp"
    match = re.match(r"^([A-Za-z]+)_([A-Za-z]+)", stem)
    if match:
        return f"{match.group(1).upper()}_{match.group(2).upper()}"
    alpha_prefix = re.match(r"^([A-Za-z]+)", stem)
    if alpha_prefix:
        return alpha_prefix.group(1).upper()
    return "other"


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _score_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": min(values, default=0.0),
        "q10": _quantile(values, 0.10),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "q90": _quantile(values, 0.90),
        "maximum": max(values, default=0.0),
        "mean": sum(values) / max(len(values), 1),
    }


def _metric_block(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows]
    labels = [bool(row["label"]) for row in rows]
    return {
        "videos": len(rows),
        "positives": sum(int(label) for label in labels),
        "negatives": sum(int(not label) for label in labels),
        "ranking": binary_ranking_metrics(scores, labels),
        "at_fixed_threshold": fixed_threshold_metrics(
            scores,
            labels,
            threshold=threshold,
        ),
    }


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            score_text = source["score"].strip()
            rows.append(
                {
                    **source,
                    "label": source["label"].strip().lower() == "true",
                    "score": None if not score_text else float(score_text),
                    "frame_alignment_delta": int(source["frame_alignment_delta"]),
                    "pose_frames": int(source["pose_frames"]),
                    "source_family": _source_family(source["video_path"]),
                }
            )
    return rows


def analyze(
    predictions: Path,
    *,
    threshold: float,
    top_errors: int,
) -> dict[str, Any]:
    rows = _read_predictions(predictions)
    evaluable = [row for row in rows if row["score"] is not None]
    positives = [row for row in evaluable if row["label"]]
    negatives = [row for row in evaluable if not row["label"]]
    matched = [row for row in evaluable if row["frame_alignment_delta"] == 0]
    shorter_pose = [row for row in evaluable if row["frame_alignment_delta"] < 0]

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluable:
        family_rows[row["source_family"]].append(row)

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluable:
        grouped_rows[row["group_id"]].append(row)
    paired_groups = {
        group_id: values
        for group_id, values in grouped_rows.items()
        if any(row["label"] for row in values)
        and any(not row["label"] for row in values)
    }
    correctly_ordered = 0
    ties = 0
    for values in paired_groups.values():
        positive_score = max(float(row["score"]) for row in values if row["label"])
        negative_score = max(float(row["score"]) for row in values if not row["label"])
        correctly_ordered += int(positive_score > negative_score)
        ties += int(positive_score == negative_score)
    paired_rows = [row for values in paired_groups.values() for row in values]

    false_positives = sorted(
        (
            row
            for row in negatives
            if float(row["score"]) >= threshold
        ),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    false_negatives = sorted(
        (
            row
            for row in positives
            if float(row["score"]) < threshold
        ),
        key=lambda row: float(row["score"]),
    )

    def error_item(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "source_family": row["source_family"],
            "score": row["score"],
            "video_path": row["video_path"],
            "frame_alignment_delta": row["frame_alignment_delta"],
        }

    return {
        "predictions": str(predictions.resolve()),
        "threshold": threshold,
        "total_rows": len(rows),
        "evaluable_rows": len(evaluable),
        "skipped_by_label_and_reason": dict(
            sorted(
                Counter(
                    f"{row['label_name']} | {row['reason']}"
                    for row in rows
                    if row["score"] is None
                ).items()
            )
        ),
        "skipped_data_quality": {
            "empty_pose_csv": sum(
                int(row["score"] is None and row["pose_frames"] == 0)
                for row in rows
            ),
            "pose_rows_but_no_valid_sample": sum(
                int(row["score"] is None and row["pose_frames"] > 0)
                for row in rows
            ),
        },
        "score_distributions": {
            "Fall": _score_summary([float(row["score"]) for row in positives]),
            "No_Fall": _score_summary([float(row["score"]) for row in negatives]),
        },
        "frame_alignment": {
            "exact": _metric_block(matched, threshold),
            "pose_csv_shorter_than_video": _metric_block(shorter_pose, threshold),
            "shorter_delta_minimum": min(
                (row["frame_alignment_delta"] for row in shorter_pose),
                default=0,
            ),
        },
        "source_families": {
            family: _metric_block(values, threshold)
            for family, values in sorted(family_rows.items())
        },
        "same_normalized_filename_groups": {
            "groups": len(paired_groups),
            "rows": len(paired_rows),
            "positive_score_above_negative_score": correctly_ordered,
            "ties": ties,
            "pairwise_ordering_accuracy": correctly_ordered
            / max(len(paired_groups), 1),
            "metrics_on_rows": _metric_block(paired_rows, threshold),
            "note": "Groups are based only on normalized filenames; shared provenance is not assumed.",
        },
        "top_false_positives": [
            error_item(row) for row in false_positives[:top_errors]
        ],
        "top_false_negatives": [
            error_item(row) for row in false_negatives[:top_errors]
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.top_errors < 0:
        raise ValueError("top-errors cannot be negative")
    result = analyze(
        args.predictions,
        threshold=args.threshold,
        top_errors=args.top_errors,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
