"""Render annotated demo videos from infer.py CSV outputs."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_text(
    frame: Any,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_video(
    source: Path,
    output: Path,
    timeline: list[dict[str, str]],
    video_row: dict[str, str],
    label: str,
    threshold: float,
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create {output}")
    timestamps = [float(row["timestamp_s"]) for row in timeline]
    video_score = float(video_row["video_score"])
    predicted_fall = video_score >= threshold
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / max(fps, 1e-6)
            timeline_index = bisect.bisect_right(timestamps, timestamp) - 1
            current = timeline[max(timeline_index, 0)]
            frame_score = float(current["score"])
            people = json.loads(current["people"])
            for person in people:
                box = person["bbox"]
                score = float(person["score"])
                left, top, right, bottom = (int(round(value)) for value in box)
                color = (40, 40, 235) if score >= threshold else (50, 210, 90)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                track = person.get("track_id")
                _draw_text(
                    frame,
                    f"ID {track}  fall {score:.2f}",
                    (left, max(24, top - 8)),
                    scale=max(0.45, width / 1800),
                    color=color,
                )

            overlay = frame.copy()
            banner_height = max(92, int(height * 0.15))
            banner_color = (25, 25, 150) if predicted_fall else (20, 95, 45)
            cv2.rectangle(overlay, (0, 0), (width, banner_height), banner_color, -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
            state = "FALL ALARM" if predicted_fall else "NORMAL"
            expected = "FALL" if label == "fall" else "ADL"
            scale = max(0.5, min(1.0, width / 1280))
            _draw_text(
                frame,
                f"GT {expected}  |  PRED {state}  |  VIDEO {video_score:.3f}",
                (18, int(banner_height * 0.42)),
                scale=scale,
                color=(255, 255, 255),
            )
            _draw_text(
                frame,
                f"t {timestamp:5.2f}s  frame {frame_score:.3f}  threshold {threshold:.4f}",
                (18, int(banner_height * 0.78)),
                scale=scale * 0.78,
                color=(230, 240, 255),
                thickness=1,
            )
            bar_left = 18
            bar_right = min(width - 18, bar_left + max(180, int(width * 0.32)))
            bar_top = banner_height + 14
            bar_bottom = bar_top + 14
            cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (50, 50, 50), -1)
            score_right = bar_left + int((bar_right - bar_left) * frame_score)
            score_color = (40, 40, 235) if frame_score >= threshold else (50, 210, 90)
            cv2.rectangle(frame, (bar_left, bar_top), (score_right, bar_bottom), score_color, -1)
            threshold_x = bar_left + int((bar_right - bar_left) * threshold)
            cv2.line(frame, (threshold_x, bar_top - 4), (threshold_x, bar_bottom + 4), (255, 255, 255), 2)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "demo" / "source")
    parser.add_argument("--inference-dir", type=Path, default=ROOT / "demo" / "inference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "demo" / "annotated")
    parser.add_argument("--labels", type=Path, default=ROOT / "demo" / "labels.json")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    videos = {Path(row["path"]).name: row for row in _rows(args.inference_dir / "predictions.csv")}
    timelines: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows(args.inference_dir / "frames.csv"):
        timelines[Path(row["path"]).name].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, label in labels.items():
        timeline = sorted(timelines[name], key=lambda row: float(row["timestamp_s"]))
        output = args.output_dir / name
        _render_video(
            args.source_dir / name,
            output,
            timeline,
            videos[name],
            str(label),
            threshold=0.7094,
        )
        print(f"rendered={output}", flush=True)


if __name__ == "__main__":
    main()
