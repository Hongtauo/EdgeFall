from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from edgefall_st.pseudo.batched_tracker import (
    BatchVideoTask,
    track_video_batch,
)
from edgefall_st.pseudo.yolo26_tracker import TeacherConfig


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.asarray([[8.0, 6.0, 40.0, 36.0]])
        self.conf = np.asarray([0.9])

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeYolo:
    def __init__(self) -> None:
        self.model = SimpleNamespace(end2end=True)
        self.batch_sizes: list[int] = []

    def predict(self, *, source, **_kwargs):
        self.batch_sizes.append(len(source))
        return [SimpleNamespace(boxes=FakeBoxes()) for _ in source]


class FakeTracker:
    def __init__(self) -> None:
        self.updates = 0

    def update(self, boxes, _frame):
        self.updates += 1
        xyxy = boxes.xyxy[0]
        return np.asarray([[*xyxy, 73.0, boxes.conf[0], 0.0, 0.0]])


def write_video(path: Path, frame_count: int, value: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


def test_batch_detection_keeps_trackers_independent(tmp_path: Path) -> None:
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    write_video(first_video, frame_count=3, value=20)
    write_video(second_video, frame_count=2, value=80)
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    trackers: list[FakeTracker] = []

    def tracker_factory(_config: TeacherConfig) -> FakeTracker:
        tracker = FakeTracker()
        trackers.append(tracker)
        return tracker

    model = FakeYolo()
    qualities = track_video_batch(
        model,
        [
            BatchVideoTask(first_video, first_output, "folder/first"),
            BatchVideoTask(second_video, second_output, "folder/second"),
        ],
        config=TeacherConfig(image_size=(48, 64)),
        decode_workers=2,
        tracker_factory=tracker_factory,
    )

    assert model.batch_sizes == [2, 2, 1]
    assert [tracker.updates for tracker in trackers] == [3, 2]
    assert [quality.processed_frames for quality in qualities] == [3, 2]
    assert all(quality.unique_track_ids == 1 for quality in qualities)
    first_rows = [
        json.loads(line)
        for line in first_output.read_text(encoding="utf-8").splitlines()
    ]
    second_rows = [
        json.loads(line)
        for line in second_output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(first_rows) == 3
    assert len(second_rows) == 2
    assert all(row["detections"][0]["track_id"] == 1 for row in first_rows)
    assert all(row["detections"][0]["track_id"] == 1 for row in second_rows)
    assert not list(tmp_path.glob("*.tmp"))
