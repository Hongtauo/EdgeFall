from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from edgefall_st.pseudo.yolo26_tracker import render_track_visualization


def test_render_track_visualization(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))
    writer.release()

    tracks_path = tmp_path / "tracks.jsonl"
    rows = [
        {
            "frame_index": frame_index,
            "timestamp_s": frame_index / 5.0,
            "action_label": 1,
            "detections": [
                {
                    "track_id": 3,
                    "bbox_xyxy": [8.0, 8.0, 40.0, 40.0],
                    "confidence": 0.9,
                }
            ],
        }
        for frame_index in range(2)
    ]
    tracks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    actor_targets_path = tmp_path / "actors.jsonl"
    actor_targets_path.write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "actor_detection_index": 0,
                }
            )
            + "\n"
            for frame_index in range(2)
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "rendered.mp4"
    rendered = render_track_visualization(
        video_path,
        tracks_path,
        output_path,
        actor_targets_path,
    )

    assert rendered == 2
    capture = cv2.VideoCapture(str(output_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()
