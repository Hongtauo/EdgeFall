from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from edgefall_st.pseudo.actor_targets import (
    ActorSelectionConfig,
    build_split_actor_targets,
    select_primary_actor,
)


def _detection(
    track_id: int,
    box: list[float],
    confidence: float = 0.9,
) -> dict[str, object]:
    left, top, right, bottom = box
    return {
        "track_id": track_id,
        "bbox_xyxy_norm": box,
        "bbox_xyxy": [
            left * 100,
            top * 100,
            right * 100,
            bottom * 100,
        ],
        "confidence": confidence,
        "area_ratio": (right - left) * (bottom - top),
    }


def _frame(
    index: int,
    detections: list[dict[str, object]],
    *,
    action_label: int = 1,
    path: str = "fall/example",
) -> dict[str, object]:
    return {
        "path": path,
        "frame_index": index,
        "timestamp_s": index / 10.0,
        "width": 100,
        "height": 100,
        "action_label": action_label,
        "detections": detections,
    }


def test_single_actor_builds_action_stage_and_onset_targets() -> None:
    frames = [
        _frame(
            0,
            [_detection(7, [0.40, 0.10, 0.60, 0.90])],
            action_label=8,
        ),
        _frame(
            1,
            [_detection(7, [0.38, 0.20, 0.68, 0.85])],
            action_label=1,
        ),
        _frame(
            2,
            [_detection(7, [0.25, 0.55, 0.80, 0.85])],
            action_label=2,
        ),
    ]

    targets, quality = select_primary_actor(frames)

    assert [target["actor_detection_index"] for target in targets] == [0, 0, 0]
    assert [target["stage_label"] for target in targets] == [0, 1, 2]
    assert [target["onset_target"] for target in targets] == [0, 1, 0]
    assert {target["actor_identity"] for target in targets} == {0}
    assert quality.confidence_level == "high"
    assert quality.action_loss_weight == 1.0


def test_foreground_actor_is_selected_and_track_id_switch_is_stitched() -> None:
    frames = []
    for frame_index in range(5):
        actor_track_id = 2 if frame_index < 3 else 9
        frames.append(
            _frame(
                frame_index,
                [
                    _detection(1, [0.70, 0.15, 0.82, 0.55]),
                    _detection(
                        actor_track_id,
                        [
                            0.15 + frame_index * 0.02,
                            0.12 + frame_index * 0.05,
                            0.62 + frame_index * 0.02,
                            0.90,
                        ],
                    ),
                ],
            )
        )

    targets, quality = select_primary_actor(frames)

    assert [target["actor_detection_index"] for target in targets] == [1] * 5
    assert quality.selected_source_track_ids == (2, 9)
    assert quality.source_track_switches == 1
    assert quality.confidence_level == "high"


def test_required_track_anchors_path_and_allows_identity_stitching() -> None:
    frames = [
        _frame(
            0,
            [
                _detection(1, [0.05, 0.10, 0.50, 0.90]),
                _detection(7, [0.60, 0.10, 0.85, 0.80]),
            ],
        ),
        _frame(
            1,
            [
                _detection(1, [0.05, 0.10, 0.50, 0.90]),
                _detection(9, [0.58, 0.12, 0.84, 0.82]),
            ],
        ),
        _frame(
            2,
            [
                _detection(1, [0.05, 0.10, 0.50, 0.90]),
                _detection(7, [0.56, 0.14, 0.83, 0.84]),
            ],
        ),
    ]

    targets, quality = select_primary_actor(frames, required_track_id=7)

    assert [target["actor_detection_index"] for target in targets] == [1, 1, 1]
    assert quality.selected_source_track_ids == (7, 9)


def test_required_track_must_exist() -> None:
    frames = [_frame(0, [_detection(1, [0.1, 0.1, 0.6, 0.9])])]

    with pytest.raises(ValueError, match="Required track ID 7 is absent"):
        select_primary_actor(frames, required_track_id=7)


def test_ambiguous_equal_sized_people_are_not_high_confidence() -> None:
    frames = [
        _frame(
            frame_index,
            [
                _detection(1, [0.08, 0.10, 0.48, 0.90]),
                _detection(2, [0.52, 0.10, 0.92, 0.90]),
            ],
        )
        for frame_index in range(5)
    ]

    _, quality = select_primary_actor(frames)

    assert quality.ambiguous_frame_ratio == 1.0
    assert quality.confidence_level == "medium"
    assert quality.action_loss_weight == pytest.approx(0.35)


def test_action_motion_overrides_a_larger_static_foreground_person() -> None:
    frames = []
    for frame_index in range(8):
        rising_top = 0.48 - 0.05 * frame_index
        rising_bottom = 0.92 - 0.035 * frame_index
        frames.append(
            _frame(
                frame_index,
                [
                    _detection(1, [0.30, 0.20, 0.75, 0.95]),
                    _detection(
                        2,
                        [0.05, rising_top, 0.32, rising_bottom],
                    ),
                ],
                action_label=7,
            )
        )
    final_actor_box = frames[-1]["detections"][1]["bbox_xyxy_norm"]
    for frame_index in range(8, 12):
        frames.append(
            _frame(
                frame_index,
                [
                    _detection(1, [0.30, 0.20, 0.75, 0.95]),
                    _detection(2, final_actor_box),
                ],
                action_label=8,
            )
        )

    targets, quality = select_primary_actor(frames)

    assert [target["actor_detection_index"] for target in targets] == [1] * 12
    assert quality.mean_action_motion > 0.7


def test_missing_actor_boxes_disable_action_and_track_losses() -> None:
    frames = [_frame(index, []) for index in range(4)]

    targets, quality = select_primary_actor(frames)

    assert quality.confidence_level == "low"
    assert quality.actor_coverage == 0.0
    assert all(target["actor_detection_index"] is None for target in targets)
    assert all(target["action_loss_weight"] == 0.0 for target in targets)
    assert all(target["track_loss_weight"] == 0.0 for target in targets)


def test_split_builder_writes_references_without_copying_detections(
    tmp_path: Path,
) -> None:
    pseudo_root = tmp_path / "pseudo"
    source_root = pseudo_root / "train_yolo26x_nms"
    source_root.mkdir(parents=True)
    with (source_root / "pilot_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split"])
        writer.writeheader()
        writer.writerow({"path": "fall/example", "split": "train"})

    track_path = source_root / "tracks" / "fall" / "example.jsonl"
    track_path.parent.mkdir(parents=True)
    frames = [
        _frame(
            index,
            [_detection(4, [0.2, 0.1, 0.7, 0.9])],
        )
        for index in range(3)
    ]
    track_path.write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames),
        encoding="utf-8",
    )

    output_root = tmp_path / "actors"
    summary = build_split_actor_targets(
        split="train",
        pseudo_root=pseudo_root,
        output_root=output_root,
        config=ActorSelectionConfig(),
        workers=1,
    )

    target_path = output_root / "train" / "targets" / "fall" / "example.jsonl"
    output_rows = [
        json.loads(line)
        for line in target_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["processed_videos"] == 1
    assert len(output_rows) == 3
    assert "detections" not in output_rows[0]
    assert output_rows[0]["actor_detection_index"] == 0
    assert (output_root / "train" / "actor_manifest.csv").exists()
