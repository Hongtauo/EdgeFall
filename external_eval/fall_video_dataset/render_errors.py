"""Render temporal storyboards for external-test false positives/negatives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from external_eval.fall_video_dataset.common import read_pose_boxes
from external_eval.fall_video_dataset.evaluate_v3_yolo import (
    read_actor_box_sequence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=5)
    return parser


def _read_errors(
    path: Path,
    *,
    threshold: float,
    examples: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "ok" and row["score"].strip()
        ]
    false_positives = sorted(
        (
            row
            for row in rows
            if row["label"].lower() == "false"
            and float(row["score"]) >= threshold
        ),
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:examples]
    false_negatives = sorted(
        (
            row
            for row in rows
            if row["label"].lower() == "true"
            and float(row["score"]) < threshold
        ),
        key=lambda row: float(row["score"]),
    )[:examples]
    return false_positives, false_negatives


def _fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    source_h, source_w = frame.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    width = max(1, round(source_w * scale))
    height = max(1, round(source_h * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 32, dtype=np.uint8)
    left = (target_w - width) // 2
    top = (target_h - height) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas


def _read_normalized_boxes(
    path: Path,
    *,
    frame_width: int,
    frame_height: int,
    frame_count: int,
) -> dict[int, tuple[float, float, float, float]]:
    if path.suffix.lower() == ".jsonl":
        return read_actor_box_sequence(
            path,
            frame_count=frame_count,
        ).boxes
    pose = read_pose_boxes(
        path,
        frame_width=frame_width,
        frame_height=frame_height,
        confidence_threshold=0.25,
        minimum_keypoints=4,
    )
    return {frame_index - 1: box for frame_index, box in pose.boxes.items()}


def _storyboard(
    rows: Sequence[dict[str, str]],
    *,
    title: str,
    frames_per_video: int,
) -> np.ndarray:
    tile_size = (256, 144)
    header_height = 38
    title_height = 44
    canvas = np.full(
        (
            title_height + len(rows) * (header_height + tile_size[1]),
            frames_per_video * tile_size[0],
            3,
        ),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for row_index, row in enumerate(rows):
        video_path = Path(row["video_path"])
        keypoints_path = Path(row["keypoints_path"])
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        boxes = _read_normalized_boxes(
            keypoints_path,
            frame_width=width,
            frame_height=height,
            frame_count=frame_count,
        )
        indices = [
            round(index * max(frame_count - 1, 0) / max(frames_per_video - 1, 1))
            for index in range(frames_per_video)
        ]
        row_top = title_height + row_index * (header_height + tile_size[1])
        cv2.putText(
            canvas,
            f"{row['sample_id']}  score={float(row['score']):.3f}",
            (8, row_top + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        for column, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
            box = boxes.get(frame_index)
            if box is not None:
                left, top, right, bottom = box
                cv2.rectangle(
                    frame,
                    (round(left * width), round(top * height)),
                    (round(right * width), round(bottom * height)),
                    (0, 255, 255),
                    max(2, round(min(width, height) / 300)),
                )
            tile = _fit_frame(frame, tile_size)
            cv2.putText(
                tile,
                f"{frame_index + 1}/{frame_count}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            left = column * tile_size[0]
            top = row_top + header_height
            canvas[top : top + tile_size[1], left : left + tile_size[0]] = tile
        capture.release()
    return canvas


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    false_positives, false_negatives = _read_errors(
        args.predictions,
        threshold=args.threshold,
        examples=args.examples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            "external_false_positives.jpg",
            _storyboard(
                false_positives,
                title="Highest-score external false positives",
                frames_per_video=args.frames_per_video,
            ),
        ),
        (
            "external_false_negatives.jpg",
            _storyboard(
                false_negatives,
                title="Lowest-score external false negatives",
                frames_per_video=args.frames_per_video,
            ),
        ),
    )
    for name, image in outputs:
        path = args.output_dir / name
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
        print(path)


if __name__ == "__main__":
    main()
