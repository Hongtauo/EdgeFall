from __future__ import annotations

import json
from pathlib import Path

import torch

from edgefall_st.features.mine_hard_negatives import (
    _ScoredWindow,
    _build_candidates,
)
from edgefall_st.training.bbox_criterion import (
    BBoxClassifierLossWeights,
    BBoxFallCriterion,
)
from edgefall_st.training.hard_negatives import load_hard_negative_registry
from edgefall_st.training.hard_positives import load_hard_positive_registry
from edgefall_st.training.sampler import VideoGroupedBatchSampler


def _window(path: str, index: int, *, action: int = 9) -> dict[str, object]:
    return {
        "path": path,
        "window_index": index,
        "end_time_s": index * 0.1,
        "action_label": action,
    }


def test_hard_negative_registry_resolves_selected_windows(tmp_path: Path) -> None:
    windows = [_window("normal/example", index) for index in range(4)]
    manifest = tmp_path / "hard.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": "normal/example",
                "domain": "omnifall",
                "kind": "semantic",
                "category": "lie_down",
                "anchor_window_index": 2,
                "window_indices": [1, 2, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    registry = load_hard_negative_registry(
        manifest,
        windows,
        positive_paths=set(),
    )

    assert registry.semantic_anchors == {"normal/example": (2,)}
    assert registry.score_anchors == {}
    assert registry.anchor_categories == {2: "lie_down"}
    assert registry.anchor_window_categories == {
        ("normal/example", 2): "lie_down"
    }
    assert registry.window_categories == {
        ("normal/example", 1): ("lie_down",),
        ("normal/example", 2): ("lie_down",),
        ("normal/example", 3): ("lie_down",),
    }
    assert registry.window_keys == {
        ("normal/example", 1),
        ("normal/example", 2),
        ("normal/example", 3),
    }


def test_hard_negative_registry_preserves_overlapping_chunk_categories(
    tmp_path: Path,
) -> None:
    windows = [_window("normal/example", index) for index in range(5)]
    manifest = tmp_path / "hard.jsonl"
    rows = [
        {
            "path": "normal/example",
            "domain": "omnifall",
            "kind": "semantic",
            "category": "sit_down",
            "anchor_window_index": 1,
            "window_indices": [0, 1, 2],
        },
        {
            "path": "normal/example",
            "domain": "omnifall",
            "kind": "score",
            "category": "score_mined_unknown",
            "anchor_window_index": 3,
            "window_indices": [2, 3, 4],
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    registry = load_hard_negative_registry(
        manifest,
        windows,
        positive_paths=set(),
    )

    assert registry.window_categories[("normal/example", 0)] == ("sit_down",)
    assert registry.window_categories[("normal/example", 2)] == (
        "score_mined_unknown",
        "sit_down",
    )
    assert registry.window_categories[("normal/example", 4)] == (
        "score_mined_unknown",
    )


def test_sampler_injects_semantic_and_score_hard_videos() -> None:
    paths = [
        "positive",
        "ordinary_0",
        "ordinary_1",
        "ordinary_2",
        "ordinary_3",
        "semantic_0",
        "semantic_1",
        "score_0",
        "score_1",
    ]
    windows = [
        _window(
            path,
            index,
            action=(1 if path == "positive" and index < 2 else 2),
        )
        for path in paths
        for index in range(4)
    ]
    anchors = {
        str(window["path"]): index
        for index, window in enumerate(windows)
        if int(window["window_index"]) == 2
    }
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=10,
        windows_per_video=2,
        shuffle=False,
        positive_paths={"positive"},
        positive_videos_per_batch=1,
        semantic_hard_anchors={
            path: (anchors[path],) for path in ("semantic_0", "semantic_1")
        },
        score_hard_anchors={
            path: (anchors[path],) for path in ("score_0", "score_1")
        },
        hard_anchor_categories={
            anchors["semantic_0"]: "sit",
            anchors["semantic_1"]: "lie_down",
        },
        semantic_hard_videos_per_batch=1,
        score_hard_videos_per_batch=1,
        seed=5,
    )

    batches = list(sampler)

    assert batches
    for batch in batches:
        assert len(batch) == 10
        batch_paths = {str(windows[index]["path"]) for index in batch}
        assert "positive" in batch_paths
        assert batch_paths.intersection({"semantic_0", "semantic_1"})
        assert batch_paths.intersection({"score_0", "score_1"})
        assert len(batch_paths) == 5


def test_hard_positive_registry_and_sampler_replay_anchor(
    tmp_path: Path,
) -> None:
    paths = ["positive_0", "positive_1", "negative_0", "negative_1"]
    windows = [
        _window(
            path,
            index,
            action=(1 if path.startswith("positive") else 9),
        )
        for path in paths
        for index in range(4)
    ]
    manifest = tmp_path / "hard_positive.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": "positive_1",
                "domain": "omnifall",
                "anchor_window_index": 2,
                "window_indices": [1, 2],
                "video_score": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = load_hard_positive_registry(
        manifest,
        windows,
        positive_paths={"positive_0", "positive_1"},
    )
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=8,
        windows_per_video=2,
        shuffle=False,
        positive_paths={"positive_0", "positive_1"},
        positive_videos_per_batch=2,
        hard_positive_anchors=dict(registry.anchors),
        hard_positive_anchor_domains=dict(registry.anchor_domains),
        hard_positive_videos_per_batch=1,
        seed=7,
    )

    batches = list(sampler)

    assert batches
    for batch in batches:
        positive_paths = {
            str(windows[index]["path"]) for index in batch[:4]
        }
        assert "positive_1" in positive_paths


def _target(stage: int) -> dict[str, torch.Tensor]:
    return {
        "stage_label": torch.tensor(stage),
        "action_label": torch.tensor(1 if stage else 9),
        "event_target": torch.tensor(float(stage > 0)),
        "onset_target": torch.tensor(0.0),
        "event_loss_weight": torch.tensor(1.0),
        "action_loss_weight": torch.tensor(1.0),
        "control_loss_weight": torch.tensor(1.0),
    }


def test_hard_video_ranking_only_uses_registered_negative_group() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            onset=0.0,
            action=0.0,
            consistency=0.0,
            video_mil=0.0,
            video_ranking=0.0,
            hard_video_ranking=1.0,
        ),
        hard_video_ranking_margin=0.2,
        hard_negative_window_keys={("hard", 0), ("hard", 1)},
        video_pooling="asymmetric",
    )
    fall_logits = torch.tensor(
        [1.0, 1.0, 1.2, -1.0, 2.0, -2.0], requires_grad=True
    )
    outputs = {
        "stage_logits": torch.zeros(6, 3, requires_grad=True),
        "fall_logits": fall_logits,
        "onset_logits": torch.zeros(6, requires_grad=True),
        "action_logits": torch.zeros(6, 16, requires_grad=True),
    }
    targets = [_target(1), _target(1), *[_target(0) for _ in range(4)]]
    metadata = [
        {
            "path": "positive",
            "window_index": index,
            "video_has_fall_event": True,
            "event_target": True,
        }
        for index in range(2)
    ]
    metadata.extend(
        {
            "path": "hard",
            "window_index": index,
            "video_has_fall_event": False,
            "event_target": False,
        }
        for index in range(2)
    )
    metadata.extend(
        {
            "path": "ordinary",
            "window_index": index,
            "video_has_fall_event": False,
            "event_target": False,
        }
        for index in range(2)
    )

    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()

    torch.testing.assert_close(
        losses["loss_hard_video_ranking"], torch.tensor(0.4)
    )
    assert losses["num_hard_negative_groups"].item() == 1
    assert fall_logits.grad is not None
    assert fall_logits.grad[2] > 0
    assert fall_logits.grad[4] == 0


def test_candidate_builder_uses_labels_before_score_mining() -> None:
    semantic = [
        _ScoredWindow(
            path="omnifall/lie",
            domain="omnifall",
            window_index=index,
            end_time_s=index * 0.1,
            action_label=5 if 3 <= index <= 6 else 0,
            score=0.1 * index,
            label_name="",
        )
        for index in range(10)
    ]
    score_only = [
        _ScoredWindow(
            path="fall_video_dataset/no_fall",
            domain="fall_video_dataset",
            window_index=index,
            end_time_s=index * 0.1,
            action_label=9,
            score=0.05 * index,
            label_name="No_Fall",
        )
        for index in range(10)
    ]

    semantic_rows = _build_candidates(
        semantic,
        windows_per_video=4,
        semantic_count=2,
        score_count=2,
        score_threshold=0.25,
    )
    score_rows = _build_candidates(
        score_only,
        windows_per_video=4,
        semantic_count=2,
        score_count=2,
        score_threshold=0.25,
    )

    assert {row["kind"] for row in semantic_rows} == {"semantic"}
    assert {row["category"] for row in semantic_rows} == {"lie_down"}
    assert {row["kind"] for row in score_rows} == {"score"}
    assert all(len(row["window_indices"]) == 4 for row in score_rows)
