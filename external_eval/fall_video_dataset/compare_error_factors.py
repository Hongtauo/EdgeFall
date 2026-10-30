"""Compare actor-box and duration factors across TP/FP/TN/FN groups."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from external_eval.fall_video_dataset.common import read_pose_boxes
from external_eval.fall_video_dataset.evaluate_v3_yolo import (
    read_actor_box_sequence,
)


FEATURES = (
    "duration_s",
    "windows",
    "pose_coverage",
    "median_aspect",
    "maximum_aspect",
    "nonvertical_fraction",
    "median_area",
    "maximum_area",
    "center_y_range",
    "maximum_center_y_step",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def _group_name(label: bool, prediction: bool) -> str:
    if label:
        return "TP" if prediction else "FN"
    return "FP" if prediction else "TN"


def _pose_features(row: dict[str, str]) -> dict[str, float] | None:
    width = int(row["width"])
    height = int(row["height"])
    boxes_path = Path(row["keypoints_path"])
    if boxes_path.suffix.lower() == ".jsonl":
        boxes = read_actor_box_sequence(
            boxes_path,
            frame_count=int(row["source_frames"]),
        ).boxes
    else:
        boxes = read_pose_boxes(
            boxes_path,
            frame_width=width,
            frame_height=height,
            confidence_threshold=0.25,
            minimum_keypoints=4,
        ).boxes
    ordered = sorted(boxes.items())
    if not ordered:
        return None
    aspects: list[float] = []
    areas: list[float] = []
    center_y: list[tuple[int, float]] = []
    for frame_number, (left, top, right, bottom) in ordered:
        box_width = right - left
        box_height = bottom - top
        aspects.append(box_width / max(box_height, 1e-9))
        areas.append(box_width * box_height)
        center_y.append((frame_number, (top + bottom) * 0.5))
    steps = [
        abs(current_y - previous_y) / max(current_frame - previous_frame, 1)
        for (previous_frame, previous_y), (current_frame, current_y) in zip(
            center_y,
            center_y[1:],
        )
    ]
    return {
        "duration_s": int(row["source_frames"]) / float(row["source_fps"]),
        "windows": float(row["windows"]),
        "pose_coverage": int(row["valid_samples"])
        / max(int(row["effective_samples"]), 1),
        "median_aspect": statistics.median(aspects),
        "maximum_aspect": max(aspects),
        "nonvertical_fraction": sum(aspect >= 0.75 for aspect in aspects)
        / len(aspects),
        "median_area": statistics.median(areas),
        "maximum_area": max(areas),
        "center_y_range": max(value for _, value in center_y)
        - min(value for _, value in center_y),
        "maximum_center_y_step": max(steps, default=0.0),
    }


def compare(
    predictions: Path,
    *,
    threshold: float,
) -> dict[str, object]:
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or not row["score"].strip():
                continue
            label = row["label"].lower() == "true"
            prediction = float(row["score"]) >= threshold
            features = _pose_features(row)
            if features is not None:
                groups[_group_name(label, prediction)].append(features)

    result: dict[str, object] = {
        "predictions": str(predictions.resolve()),
        "threshold": threshold,
        "feature_definitions": {
            "nonvertical_fraction": "fraction of actor boxes with width/height >= 0.75",
            "maximum_center_y_step": "largest absolute normalized center-y change per source frame",
        },
        "groups": {},
    }
    output_groups = result["groups"]
    assert isinstance(output_groups, dict)
    for name, rows in sorted(groups.items()):
        output_groups[name] = {
            "count": len(rows),
            "features": {
                feature: {
                    "mean": statistics.fmean(row[feature] for row in rows),
                    "median": statistics.median(row[feature] for row in rows),
                }
                for feature in FEATURES
            },
        }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = compare(args.predictions, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
