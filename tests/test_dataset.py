from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from edgefall_st.data.dataset import (
    OmniFallWindowDataset,
    omnifall_collate_fn,
    omnifall_indexed_collate_fn,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _detection(
    track_id: int,
    box: list[float],
    *,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "bbox_xyxy_norm": box,
        "confidence": confidence,
    }


def _build_fixture(
    root: Path,
    *,
    frame_actions: list[int] | None = None,
    malformed_frame_index: bool = False,
) -> dict[str, Path]:
    sample_path = "fall/example"
    dataset_root = root / "videos"
    tracks_root = root / "tracks"
    actors_root = root / "actors"
    video_path = dataset_root / "fall" / "example.avi"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (100, 50),
    )
    assert writer.isOpened()
    for frame_index in range(6):
        image = np.full(
            (50, 100, 3),
            (20 * frame_index, 60, 120),
            dtype=np.uint8,
        )
        writer.write(image)
    writer.release()

    actions = frame_actions or [1] * 6
    tracks: list[dict[str, object]] = []
    actors: list[dict[str, object]] = []
    for frame_index in range(6):
        detections = [
            _detection(11, [0.1, 0.2, 0.5, 0.8]),
            _detection(22, [0.6, 0.0, 0.9, 1.0], confidence=0.8),
        ]
        if frame_index == 3:
            detections = detections[:1]
        stored_frame_index = (
            frame_index + 1
            if malformed_frame_index and frame_index == 3
            else frame_index
        )
        tracks.append(
            {
                "path": sample_path,
                "frame_index": stored_frame_index,
                "timestamp_s": frame_index / 10.0,
                "width": 100,
                "height": 50,
                "action_label": actions[frame_index],
                "detections": detections,
            }
        )
        actor_index = 1 if len(detections) > 1 else 0
        actors.append(
            {
                "path": sample_path,
                "frame_index": stored_frame_index,
                "timestamp_s": frame_index / 10.0,
                "action_label": actions[frame_index],
                "stage_label": 1,
                "onset_target": int(frame_index == 0),
                "actor_detection_index": actor_index,
                "actor_source_track_id": detections[actor_index]["track_id"],
                "actor_identity": 0,
                "action_loss_weight": 1.0,
                "track_loss_weight": 1.0,
            }
        )

    windows_path = root / "windows.jsonl"
    _write_jsonl(
        windows_path,
        [
            {
                "path": sample_path,
                "window_index": 123,
                "end_frame": 2,
                "end_time_s": 0.2,
                "start_time_s": 0.0,
                "left_padding": 1,
                "action_label": 1,
                "stage_label": 1,
            },
            {
                "path": sample_path,
                "end_frame": 3,
                "end_time_s": 0.3,
                "start_time_s": 0.0,
                "left_padding": 0,
                "action_label": 1,
                "stage_label": 1,
            },
        ],
    )
    _write_jsonl(tracks_root / "fall" / "example.jsonl", tracks)
    _write_jsonl(actors_root / "fall" / "example.jsonl", actors)
    return {
        "dataset_root": dataset_root,
        "windows_path": windows_path,
        "tracks_root": tracks_root,
        "actor_targets_root": actors_root,
    }


def _dataset(
    paths: dict[str, Path],
    **kwargs: object,
) -> OmniFallWindowDataset:
    return OmniFallWindowDataset(
        **paths,
        input_size=(100, 100),
        effective_fps=10.0,
        history_offsets=(0, 1, 2, 4),
        video_suffix=".avi",
        **kwargs,
    )


def test_dataset_loads_sparse_history_and_letterboxed_targets(
    tmp_path: Path,
) -> None:
    dataset = _dataset(_build_fixture(tmp_path))

    sample = dataset[0]

    assert sample["images"].shape == (4, 3, 100, 100)
    assert sample["images"].dtype == torch.float32
    assert sample["valid_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert torch.count_nonzero(sample["images"][3]) == 0
    assert sample["metadata"]["history_frame_indices"] == [2, 1, 0, -1]
    assert sample["metadata"]["source_frame_index"] == 2
    assert sample["metadata"]["action_aligned"] is True
    assert sample["metadata"]["video_has_fall_event"] is True
    assert sample["metadata"]["event_target"] is True
    assert sample["metadata"]["window_index"] == 123
    assert sample["actor_boxes"].shape == (4, 4)
    assert sample["actor_valid_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]
    torch.testing.assert_close(
        sample["actor_boxes"][0],
        torch.tensor([0.6, 0.25, 0.9, 0.75]),
    )

    target = sample["target"]
    expected_boxes = torch.tensor(
        [
            [0.1, 0.35, 0.5, 0.65],
            [0.6, 0.25, 0.9, 0.75],
        ]
    )
    torch.testing.assert_close(target["boxes"], expected_boxes)
    assert target["actor_box_index"].item() == 1
    assert target["source_track_ids"].tolist() == [11, 22]
    assert target["action_label"].item() == 1
    assert target["stage_label"].item() == 1
    assert target["event_target"].item() == 1.0
    assert target["onset_target"].item() == 1.0
    assert target["source_onset_target"].item() == 0.0
    assert target["action_loss_weight"].item() == 1.0


def test_dataset_builds_label_blind_secondary_actor_path(
    tmp_path: Path,
) -> None:
    dataset = _dataset(_build_fixture(tmp_path), actor_top_k=2)

    sample = dataset[0]

    assert sample["actor_boxes"].shape == (4, 2, 4)
    assert sample["actor_valid_mask"].shape == (4, 2)
    assert sample["actor_valid_mask"].tolist() == [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ]
    torch.testing.assert_close(
        sample["actor_boxes"][0, 0],
        torch.tensor([0.6, 0.25, 0.9, 0.75]),
    )
    torch.testing.assert_close(
        sample["actor_boxes"][0, 1],
        torch.tensor([0.1, 0.35, 0.5, 0.65]),
    )

    indexed = omnifall_indexed_collate_fn(
        dataset.__getitems__([0, 1])
    )
    assert indexed["actor_boxes"].shape == (4, 2, 4)
    assert indexed["actor_valid_mask"].shape == (4, 2)


def test_dataset_repairs_one_frame_action_boundary(tmp_path: Path) -> None:
    paths = _build_fixture(
        tmp_path,
        frame_actions=[1, 1, 8, 8, 8, 8],
    )
    dataset = _dataset(paths)

    sample = dataset[0]

    assert sample["metadata"]["source_frame_index"] == 1
    assert sample["metadata"]["source_action_label"] == 1
    assert sample["metadata"]["action_aligned"] is True


def test_unresolved_action_mismatch_disables_actor_losses(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(
        tmp_path,
        frame_actions=[8, 8, 8, 8, 8, 8],
    )
    dataset = _dataset(paths, boundary_search_radius=1)

    sample = dataset[0]

    assert sample["metadata"]["action_aligned"] is False
    assert sample["target"]["action_loss_weight"].item() == 0.0
    assert sample["target"]["track_loss_weight"].item() == 0.0


def test_collate_preserves_variable_length_targets(tmp_path: Path) -> None:
    dataset = _dataset(_build_fixture(tmp_path))

    batch = omnifall_collate_fn([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 4, 3, 100, 100)
    assert batch["valid_mask"].shape == (2, 4)
    assert batch["actor_boxes"].shape == (2, 4, 4)
    assert batch["actor_valid_mask"].shape == (2, 4)
    assert [target["boxes"].shape[0] for target in batch["targets"]] == [2, 1]
    assert len(batch["metadata"]) == 2


def test_uint8_dataset_and_indexed_collate_deduplicate_frames(
    tmp_path: Path,
) -> None:
    dataset = _dataset(_build_fixture(tmp_path), return_uint8=True)

    samples = dataset.__getitems__([0, 1])
    batch = omnifall_indexed_collate_fn(samples)

    assert samples[0]["images"].dtype == torch.uint8
    assert batch["images"].dtype == torch.uint8
    assert batch["images"].shape == (4, 3, 100, 100)
    assert batch["actor_boxes"].shape == (4, 4)
    assert batch["actor_valid_mask"].shape == (4,)
    assert batch["history_indices"].tolist() == [
        [0, 1, 2, -1],
        [3, 0, 1, -1],
    ]
    assert batch["valid_mask"].tolist() == [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ]


def test_bulk_getitems_decodes_one_pass_for_shared_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(_build_fixture(tmp_path))
    original_decode = dataset._decode_frames
    calls: list[tuple[int | None, ...]] = []

    def counted_decode(
        video_path: Path,
        positions: list[int | None],
    ) -> dict[int, np.ndarray]:
        calls.append(tuple(positions))
        return original_decode(video_path, positions)

    monkeypatch.setattr(dataset, "_decode_frames", counted_decode)

    samples = dataset.__getitems__([0, 1])

    assert len(samples) == 2
    assert len(calls) == 1
    assert set(calls[0]) == {0, 1, 2, 3}
    assert samples[0]["metadata"]["source_frame_index"] == 2
    assert samples[1]["metadata"]["source_frame_index"] == 3


def test_event_table_overrides_sparse_window_event_inference(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    events_path = paths["windows_path"].with_name("events.csv")
    events_path.write_text(
        "path,start_s,end_s,falling_end_s,includes_fallen\n"
        "fall/different,0.0,1.0,1.0,False\n",
        encoding="utf-8",
    )

    sample = _dataset(paths)[0]

    assert sample["metadata"]["video_has_fall_event"] is False
    assert sample["metadata"]["event_target"] is False
    assert sample["target"]["event_target"].item() == 0.0


def test_event_target_uses_the_causal_event_interval(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    events_path = paths["windows_path"].with_name("events.csv")
    events_path.write_text(
        "path,start_s,end_s,falling_end_s,includes_fallen\n"
        "fall/example,0.25,0.4,0.4,False\n",
        encoding="utf-8",
    )

    dataset = _dataset(paths)
    before_event = dataset[0]
    inside_event = dataset[1]

    assert before_event["metadata"]["video_has_fall_event"] is True
    assert before_event["metadata"]["event_target"] is False
    assert before_event["target"]["event_target"].item() == 0.0
    assert inside_event["metadata"]["event_target"] is True
    assert inside_event["target"]["event_target"].item() == 1.0


def test_dataset_reads_precomputed_actor_crops_without_video_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    crop_root = tmp_path / "crops"
    crop_path = crop_root / "fall" / "example.npy"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crops = np.arange(6 * 3 * 16 * 16, dtype=np.uint8).reshape(
        6,
        3,
        16,
        16,
    )
    np.save(crop_path, crops, allow_pickle=False)
    dataset = _dataset(
        paths,
        actor_crops_root=crop_root,
        actor_crop_size=(16, 16),
    )

    def fail_decode(*args: object, **kwargs: object) -> dict[int, np.ndarray]:
        del args, kwargs
        raise AssertionError("Video decoding must not run for cached crops")

    monkeypatch.setattr(dataset, "_decode_frames", fail_decode)
    sample = dataset[0]

    assert sample["images"].shape == (4, 3, 16, 16)
    assert sample["metadata"]["pre_cropped_input"] is True
    assert sample["metadata"]["frame_size"] == (16, 16)
    torch.testing.assert_close(
        sample["images"][0],
        torch.from_numpy(crops[2]).float().div(255.0),
    )


def test_missing_actor_box_disables_actor_losses(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    actor_path = paths["actor_targets_root"] / "fall" / "example.jsonl"
    actors = [
        json.loads(line)
        for line in actor_path.read_text(encoding="utf-8").splitlines()
    ]
    actors[2]["actor_detection_index"] = 99
    _write_jsonl(actor_path, actors)
    dataset = _dataset(paths)

    target = dataset[0]["target"]

    assert target["actor_box_index"].item() == -1
    assert target["action_loss_weight"].item() == 0.0
    assert target["track_loss_weight"].item() == 0.0


def test_malformed_track_indices_are_rejected(tmp_path: Path) -> None:
    dataset = _dataset(
        _build_fixture(tmp_path, malformed_frame_index=True)
    )

    with pytest.raises(ValueError, match="Non-contiguous track frame"):
        _ = dataset[0]


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        omnifall_collate_fn([])
    with pytest.raises(ValueError, match="empty batch"):
        omnifall_indexed_collate_fn([])
