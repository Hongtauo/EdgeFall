"""Dataset discovery, pose parsing, sampling, and binary metrics."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LABEL_DIRECTORIES = {"Fall": True, "No_Fall": False}
REQUIRED_POSE_COLUMNS = {"Frame", "Keypoint", "X", "Y", "Confidence"}


class DatasetLayoutError(RuntimeError):
    """Raised when the external dataset does not match its expected layout."""


@dataclass(frozen=True)
class VideoRecord:
    sample_id: str
    group_id: str
    label_name: str
    label: bool
    video_path: Path
    keypoints_path: Path


@dataclass(frozen=True)
class PoseBoxData:
    boxes: dict[int, tuple[float, float, float, float]]
    rows: int
    observed_frames: int
    valid_frames: int
    maximum_frame_number: int


def normalized_group_id(stem: str) -> str:
    """Group class-specific clips that appear to share an original source."""

    return re.sub(r"_resized$", "", stem, flags=re.IGNORECASE).casefold()


def discover_records(dataset_root: str | Path) -> list[VideoRecord]:
    root = Path(dataset_root)
    records: list[VideoRecord] = []
    problems: list[str] = []
    for label_name, label in LABEL_DIRECTORIES.items():
        class_root = root / label_name
        video_root = class_root / "Raw_Video"
        keypoints_root = class_root / "Keypoints_CSV"
        if not video_root.is_dir() or not keypoints_root.is_dir():
            problems.append(
                f"{label_name}: expected Raw_Video and Keypoints_CSV"
            )
            continue
        videos = {path.stem: path for path in video_root.glob("*.mp4")}
        keypoints = {
            path.stem.removesuffix("_keypoints"): path
            for path in keypoints_root.glob("*.csv")
        }
        missing_keypoints = sorted(set(videos) - set(keypoints))
        missing_videos = sorted(set(keypoints) - set(videos))
        if missing_keypoints:
            problems.append(
                f"{label_name}: {len(missing_keypoints)} videos lack CSV files"
            )
        if missing_videos:
            problems.append(
                f"{label_name}: {len(missing_videos)} CSV files lack videos"
            )
        for stem in sorted(set(videos) & set(keypoints)):
            records.append(
                VideoRecord(
                    sample_id=f"{label_name}/{stem}",
                    group_id=normalized_group_id(stem),
                    label_name=label_name,
                    label=label,
                    video_path=videos[stem],
                    keypoints_path=keypoints[stem],
                )
            )
    if problems:
        raise DatasetLayoutError("; ".join(problems))
    if not records:
        raise DatasetLayoutError(f"No video/CSV pairs found under {root}")
    return records


def read_pose_boxes(
    path: str | Path,
    *,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float = 0.25,
    minimum_keypoints: int = 4,
) -> PoseBoxData:
    """Convert one-person COCO keypoints into normalized tight boxes.

    CSV frame numbers are retained as one-based values so callers can compare
    them directly with the source annotation. Visual context expansion remains
    the responsibility of the model's ActorCropper.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if minimum_keypoints <= 0:
        raise ValueError("minimum_keypoints must be positive")

    grouped_points: dict[int, list[tuple[float, float]]] = {}
    observed_frames: set[int] = set()
    rows = 0
    maximum_frame_number = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_POSE_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Unexpected keypoint CSV columns in {path}")
        for row in reader:
            rows += 1
            try:
                frame_number = int(row["Frame"])
                x = float(row["X"])
                y = float(row["Y"])
                confidence = float(row["Confidence"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid keypoint row {rows} in {path}") from error
            if frame_number <= 0:
                raise ValueError(f"Frame numbers must be one-based in {path}")
            observed_frames.add(frame_number)
            maximum_frame_number = max(maximum_frame_number, frame_number)
            if (
                confidence < confidence_threshold
                or not math.isfinite(x)
                or not math.isfinite(y)
            ):
                continue
            grouped_points.setdefault(frame_number, []).append(
                (
                    min(max(x, 0.0), float(frame_width)),
                    min(max(y, 0.0), float(frame_height)),
                )
            )

    boxes: dict[int, tuple[float, float, float, float]] = {}
    for frame_number, points in grouped_points.items():
        if len(points) < minimum_keypoints:
            continue
        xs, ys = zip(*points, strict=True)
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        if right - left < 2.0 or bottom - top < 2.0:
            continue
        boxes[frame_number] = (
            left / frame_width,
            top / frame_height,
            right / frame_width,
            bottom / frame_height,
        )
    return PoseBoxData(
        boxes=boxes,
        rows=rows,
        observed_frames=len(observed_frames),
        valid_frames=len(boxes),
        maximum_frame_number=maximum_frame_number,
    )


def resampled_source_indices(
    frame_count: int,
    source_fps: float,
    target_fps: float,
) -> tuple[int, ...]:
    """Map an effective fixed-rate timeline to zero-based source frames."""

    if frame_count <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("frame_count and frame rates must be positive")
    duration_s = (frame_count - 1) / source_fps
    sample_count = int(math.floor(duration_s * target_fps + 1e-9)) + 1
    return tuple(
        min(frame_count - 1, int(round(index * source_fps / target_fps)))
        for index in range(sample_count)
    )


def binary_ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> dict[str, float]:
    """Match the AP/P90/P95 calculation used by V3 training."""

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    empty = {
        "average_precision": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "mean_precision": 0.0,
    }
    if not scores:
        return empty
    records = sorted(
        zip(scores, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    positive_count = sum(int(label) for label in labels)
    if positive_count == 0:
        return empty

    curve: list[tuple[float, float]] = []
    true_positives = 0
    false_positives = 0
    average_precision = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(records):
        score = records[index][0]
        group_true_positives = 0
        group_size = 0
        while index < len(records) and records[index][0] == score:
            group_true_positives += int(records[index][1])
            group_size += 1
            index += 1
        true_positives += group_true_positives
        false_positives += group_size - group_true_positives
        recall = true_positives / positive_count
        precision = true_positives / max(true_positives + false_positives, 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        curve.append((recall, precision))

    def precision_at_recall(target: float) -> float:
        return max(
            (precision for recall, precision in curve if recall >= target),
            default=0.0,
        )

    p90 = precision_at_recall(0.90)
    p95 = precision_at_recall(0.95)
    return {
        "average_precision": average_precision,
        "p90": p90,
        "p95": p95,
        "mean_precision": (p90 + p95) * 0.5,
    }


def fixed_threshold_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    true_positives = false_positives = true_negatives = false_negatives = 0
    for score, label in zip(scores, labels, strict=True):
        prediction = score >= threshold
        true_positives += int(prediction and label)
        false_positives += int(prediction and not label)
        true_negatives += int(not prediction and not label)
        false_negatives += int(not prediction and label)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    specificity = true_negatives / max(true_negatives + false_positives, 1)
    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "accuracy": (true_positives + true_negatives) / max(len(scores), 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }
