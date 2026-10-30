from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from edgefall_st.features.adapter import FrozenPyramidFeatureAdapter
from edgefall_st.features.cache import (
    FrozenFeatureCacheWriter,
    FrozenFeatureWindowDataset,
    MultiDomainFrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
)
from edgefall_st.features.extract import _remap_letterboxed_boxes
from edgefall_st.features.extract_multidomain import (
    CacheVideo,
    _boxes_path,
)
from edgefall_st.features.mine import run_mining
from edgefall_st.evaluation.fygr import _end_to_end_video_metrics
from edgefall_st.evaluation.select_fygr_checkpoint import _domain_caches
from edgefall_st.models.bbox_classifier import BBoxFallClassifier
from edgefall_st.models.fygr import (
    FYGR_FUSION_CROSS_ATTENTION,
    FYGR_FUSION_TEMPORAL_TOKENS,
    FrozenYoloGlobalROIConfirmer,
)
from edgefall_st.runtime.fygr_online import (
    CausalActorSelector,
    TrackedPerson,
    VideoOnlineResult,
    _ranking_metrics,
    _stratified_sample,
)
from edgefall_st.training.bbox_criterion import (
    BBoxClassifierLossWeights,
    BBoxFallCriterion,
)
from edgefall_st.training.fygr_criterion import FYGRCriterion
from edgefall_st.training.sampler import VideoGroupedBatchSampler
from edgefall_st.training.train_fygr import (
    _FusionStabilityAccumulator,
    _HardTailMetricAccumulator,
    _model_version,
    _optimizer,
    _selection_metric_name,
    _validate,
    build_parser,
)


HISTORY_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32)
VISUAL_OFFSETS = (0, 2, 4, 7, 8, 16, 24, 32)


def test_online_ranking_metrics_are_standalone() -> None:
    results = [
        VideoOnlineResult(
            path=f"video_{index}",
            label=label,
            source_fps=10.0,
            source_frames=10,
            sampled_frames=10,
            video_score=score,
            first_alarm_s=None,
            decode_ms=0.0,
            wall_s=0.0,
        )
        for index, (score, label) in enumerate(
            ((0.9, 1), (0.8, 0), (0.7, 1), (0.1, 0))
        )
    ]
    metrics = _ranking_metrics(results)
    assert metrics["average_precision"] == pytest.approx(5.0 / 6.0)
    assert metrics["p90"] == pytest.approx(2.0 / 3.0)
    assert metrics["p95"] == pytest.approx(2.0 / 3.0)
    assert metrics["mean_precision"] == pytest.approx(2.0 / 3.0)


class _FakeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p3 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
        self.p4 = nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1)
        self.p5 = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        p3 = self.p3(images)
        p4 = self.p4(p3)
        return self.p5(p4)


def test_frozen_adapter_contract_and_invalid_roi() -> None:
    detector = _FakeDetector()
    adapter = FrozenPyramidFeatureAdapter(
        detector,
        feature_modules={
            "p3": detector.p3,
            "p4": detector.p4,
            "p5": detector.p5,
        },
        roi_output_size=3,
    )
    output = adapter(
        torch.rand(2, 3, 64, 96),
        torch.tensor(
            [[0.1, 0.2, 0.6, 0.9], [0.0, 0.0, 0.0, 0.0]]
        ),
        torch.tensor([1.0, 0.0]),
    )
    assert output.global_features.shape == (2, 14)
    assert output.roi_features.shape == (2, 10, 3, 3)
    assert torch.count_nonzero(output.roi_features[1]) == 0
    assert all(not parameter.requires_grad for parameter in detector.parameters())
    assert adapter.feature_spec()["channels"] == {"p3": 4, "p4": 6, "p5": 8}


def test_frozen_adapter_reuses_captured_detector_features() -> None:
    detector = _FakeDetector()
    adapter = FrozenPyramidFeatureAdapter(
        detector,
        feature_modules={
            "p3": detector.p3,
            "p4": detector.p4,
            "p5": detector.p5,
        },
    )
    adapter.clear_captured_features()
    detector(torch.rand(1, 3, 64, 96))
    output = adapter.features_from_captured(
        torch.tensor([[0.1, 0.2, 0.6, 0.9]]),
        torch.ones(1),
    )
    assert output.global_features.shape == (1, 14)
    assert output.roi_features.shape == (1, 10, 3, 3)


def test_frozen_adapter_pools_multiple_people_from_one_forward() -> None:
    detector = _FakeDetector()
    adapter = FrozenPyramidFeatureAdapter(
        detector,
        feature_modules={
            "p3": detector.p3,
            "p4": detector.p4,
            "p5": detector.p5,
        },
    )
    output = adapter(
        torch.rand(2, 3, 64, 96),
        torch.tensor(
            [
                [[0.1, 0.1, 0.4, 0.8], [0.5, 0.2, 0.9, 0.9]],
                [[0.2, 0.2, 0.7, 0.9], [0.0, 0.0, 0.0, 0.0]],
            ]
        ),
        torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    )
    assert output.global_features.shape == (2, 14)
    assert output.roi_features.shape == (2, 2, 10, 3, 3)
    assert torch.count_nonzero(output.roi_features[1, 1]) == 0


def test_causal_actor_selector_preserves_track_and_waits_before_switch() -> None:
    selector = CausalActorSelector(max_missed=1)
    first = TrackedPerson((10.0, 10.0, 80.0, 90.0), 0.8, 7)
    distractor = TrackedPerson((40.0, 20.0, 70.0, 80.0), 0.95, 8)
    assert selector.select([first, distractor], (100, 100, 3)) == first
    assert selector.select([distractor], (100, 100, 3)) is None
    assert selector.select([distractor], (100, 100, 3)) == distractor


def test_online_manifest_sampling_is_deterministic_and_stratified() -> None:
    rows = [
        {"path": f"negative/{index}", "label": "0"}
        for index in range(8)
    ] + [
        {"path": f"positive/{index}", "label": "1"}
        for index in range(2)
    ]
    first = _stratified_sample(rows, 5, seed=7)
    second = _stratified_sample(rows, 5, seed=7)
    assert first == second
    assert len(first) == 5
    assert sum(int(row["label"]) for row in first) == 1


def test_letterbox_box_remapping_round_trip() -> None:
    boxes = torch.tensor(
        [[[0.2, 0.25, 0.7, 0.9], [0.1, 0.1, 0.3, 0.4]]]
    )
    mapped = _remap_letterboxed_boxes(
        boxes,
        [(720, 1280)],
        source_input_size=(544, 960),
        target_input_size=(288, 512),
    )
    restored = _remap_letterboxed_boxes(
        mapped,
        [(720, 1280)],
        source_input_size=(288, 512),
        target_input_size=(544, 960),
    )
    assert torch.allclose(restored, boxes, atol=1e-5)


def test_multidomain_boxes_path_prefers_current_and_supports_legacy(
    tmp_path: Path,
) -> None:
    record = CacheVideo(
        path="fall_video_dataset/Fall/example",
        domain="fall_video_dataset",
        split="train",
        label=True,
        label_name="Fall",
        video_path=tmp_path / "video.avi",
    )
    args = SimpleNamespace(boxes_root=tmp_path)
    current = (
        tmp_path
        / "actor_boxes"
        / "fall_video_dataset"
        / "Fall"
        / "example.jsonl"
    )
    current.parent.mkdir(parents=True)
    current.touch()
    assert _boxes_path(args, record) == current
    current.unlink()
    assert _boxes_path(args, record) == (
        tmp_path / "actor_boxes" / "Fall" / "example.jsonl"
    )


def test_end_to_end_video_metrics_assign_zero_to_missing_video(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "domain,split,path,label\n"
        "example,test,positive,1\n"
        "example,test,negative,0\n"
        "example,test,missing_positive,1\n",
        encoding="utf-8",
    )
    records = [
        {
            "path": "positive",
            "score": 0.9,
            "geometry_score": 0.8,
            "global_score": 0.7,
            "roi_score": 0.6,
        },
        {
            "path": "negative",
            "score": 0.2,
            "geometry_score": 0.1,
            "global_score": 0.3,
            "roi_score": 0.4,
        },
    ]
    metrics, missing = _end_to_end_video_metrics(
        records,
        manifest_path=manifest,
        domain="example",
        split="test",
    )
    assert metrics["end_to_end_expected_videos"] == 3
    assert metrics["end_to_end_missing_positive_videos"] == 1
    assert metrics["end_to_end_video_coverage"] == 2 / 3
    assert missing == [
        {"path": "missing_positive", "label": 1, "assigned_score": 0.0}
    ]


def test_end_to_end_video_metrics_accepts_fused_score_only(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "domain,split,path,label\n"
        "example,val,positive,1\n"
        "example,val,negative,0\n",
        encoding="utf-8",
    )
    metrics, missing = _end_to_end_video_metrics(
        [
            {"path": "positive", "score": 0.9},
            {"path": "negative", "score": 0.1},
        ],
        manifest_path=manifest,
        domain="example",
        split="val",
    )
    assert metrics["end_to_end_video_mean_precision"] == 1.0
    assert missing == []


def test_checkpoint_selection_parses_unique_domain_caches() -> None:
    assert _domain_caches(["omnifall=cache/val"]) == [
        ("omnifall", Path("cache/val"))
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        _domain_caches(["omnifall=one", "omnifall=two"])
    with pytest.raises(ValueError, match="DOMAIN=PATH"):
        _domain_caches(["invalid"])


def _cache_record(index: int) -> dict[str, object]:
    return {
        "path": "video/sample",
        "video_has_fall_event": True,
        "event_target": bool(index),
        "window_index": index,
        "end_frame": 10 + index,
        "end_time_s": 1.0 + index,
        "window_action_label": 1 if index else 0,
        "action_label": 1 if index else 0,
        "stage_label": 1 if index else 0,
        "onset_target": float(index),
        "action_loss_weight": 1.0,
        "history_frame_indices": list(HISTORY_OFFSETS),
        "source_size": [720, 1280],
    }


def test_feature_cache_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    writer = FrozenFeatureCacheWriter(
        root,
        metadata={
            "history_offsets": list(HISTORY_OFFSETS),
            "visual_offsets": list(VISUAL_OFFSETS),
        },
    )
    writer.write_video(
        "video/sample",
        frame_indices=np.asarray(HISTORY_OFFSETS),
        visual_frame_indices=np.asarray(VISUAL_OFFSETS),
        actor_boxes=np.ones((15, 4), dtype=np.float32),
        actor_valid_mask=np.ones(15, dtype=np.float32),
        visual_actor_boxes=np.ones((8, 4), dtype=np.float32),
        global_features=np.ones((8, 5), dtype=np.float16),
        roi_features=np.ones((8, 4, 2, 2), dtype=np.float16),
        source_size=(720, 1280),
        records=[_cache_record(0), _cache_record(1)],
    )
    writer.close()
    dataset = FrozenFeatureWindowDataset(root)
    batch = frozen_feature_collate_fn([dataset[0], dataset[1]])
    assert batch["global_features"].shape == (2, 8, 5)
    assert batch["roi_features"].shape == (2, 8, 4, 2, 2)
    assert len(batch["targets"]) == 2


def test_multi_person_feature_cache_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    writer = FrozenFeatureCacheWriter(
        root,
        metadata={
            "history_offsets": list(HISTORY_OFFSETS),
            "visual_offsets": list(VISUAL_OFFSETS),
        },
    )
    writer.write_video(
        "video/sample",
        frame_indices=np.asarray(HISTORY_OFFSETS),
        visual_frame_indices=np.asarray(VISUAL_OFFSETS),
        actor_boxes=np.ones((15, 2, 4), dtype=np.float32),
        actor_valid_mask=np.ones((15, 2), dtype=np.float32),
        visual_actor_boxes=np.ones((8, 2, 4), dtype=np.float32),
        global_features=np.ones((8, 5), dtype=np.float16),
        roi_features=np.ones((8, 2, 4, 2, 2), dtype=np.float16),
        source_size=(720, 1280),
        records=[_cache_record(0), _cache_record(1)],
    )
    writer.close()
    dataset = FrozenFeatureWindowDataset(root)
    batch = frozen_feature_collate_fn([dataset[0], dataset[1]])
    assert batch["roi_features"].shape == (2, 8, 2, 4, 2, 2)
    assert batch["actor_boxes"].shape == (2, 15, 2, 4)
    assert batch["actor_valid_mask"].shape == (2, 15, 2)


def test_multidomain_cache_preserves_domain_and_weak_weights(
    tmp_path: Path,
) -> None:
    datasets = []
    for domain, path, weak in (
        ("omnifall", "video/source", False),
        ("fall_video_dataset", "fall_video_dataset/Fall/example", True),
    ):
        root = tmp_path / domain
        writer = FrozenFeatureCacheWriter(
            root,
            metadata={
                "domain": domain,
                "history_offsets": list(HISTORY_OFFSETS),
                "visual_offsets": list(VISUAL_OFFSETS),
            },
        )
        record = {
            **_cache_record(1),
            "path": path,
            "domain": domain,
            "weak_video_label": weak,
            "event_loss_weight": 0.0 if weak else 1.0,
            "control_loss_weight": 0.0 if weak else 1.0,
        }
        writer.write_video(
            path,
            frame_indices=np.asarray(HISTORY_OFFSETS),
            visual_frame_indices=np.asarray(VISUAL_OFFSETS),
            actor_boxes=np.ones((15, 4), dtype=np.float32),
            actor_valid_mask=np.ones(15, dtype=np.float32),
            visual_actor_boxes=np.ones((8, 4), dtype=np.float32),
            global_features=np.ones((8, 5), dtype=np.float16),
            roi_features=np.ones((8, 4, 2, 2), dtype=np.float16),
            source_size=(720, 1280),
            records=[record],
        )
        writer.close()
        datasets.append(FrozenFeatureWindowDataset(root))
    combined = MultiDomainFrozenFeatureWindowDataset(datasets)
    assert len(combined) == 2
    assert combined[1]["metadata"]["domain"] == "fall_video_dataset"
    assert combined[1]["metadata"]["weak_video_label"] is True
    assert float(combined[1]["target"]["event_loss_weight"]) == 0.0
    assert combined.weak_positive_paths == {
        "fall_video_dataset/Fall/example"
    }


def test_weak_positive_video_uses_mil_without_window_label() -> None:
    base = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            event_state=0.0,
            onset=0.0,
            action=0.0,
            control=0.0,
            consistency=0.0,
            video_mil=1.0,
            video_ranking=0.0,
        ),
        video_pooling="asymmetric",
        video_topk=2,
    )
    criterion = FYGRCriterion(
        base,
        global_aux_weight=1.0,
        roi_aux_weight=0.0,
        weak_video_aux_weight=1.0,
    )
    fall_logits = torch.tensor([-1.0, 1.0], requires_grad=True)
    global_logits = torch.tensor([-0.5, 0.5], requires_grad=True)
    outputs = {
        "stage_logits": torch.zeros(2, 3),
        "fall_logits": fall_logits,
        "onset_logits": torch.zeros(2),
        "action_logits": torch.zeros(2, 16),
        "global_fall_logits": global_logits,
    }
    targets = [
        {
            "stage_label": torch.tensor(0),
            "action_label": torch.tensor(9),
            "event_target": torch.tensor(0.0),
            "onset_target": torch.tensor(0.0),
            "event_loss_weight": torch.tensor(0.0),
            "action_loss_weight": torch.tensor(0.0),
            "control_loss_weight": torch.tensor(0.0),
        }
        for _ in range(2)
    ]
    metadata = [
        {
            "path": "fall_video_dataset/Fall/example",
            "video_has_fall_event": True,
            "weak_video_label": True,
            "event_target": False,
        }
        for _ in range(2)
    ]
    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()
    assert float(losses["loss_video_mil"].detach()) > 0.0
    assert float(losses["loss_global_aux_video_mil"].detach()) > 0.0
    assert fall_logits.grad is not None
    assert global_logits.grad is not None


def test_sampler_pads_short_weak_positive_video() -> None:
    windows = [
        {"path": "positive", "end_time_s": 0.0, "action_label": 9},
        {"path": "negative", "end_time_s": 0.0, "action_label": 9},
        {"path": "negative", "end_time_s": 0.1, "action_label": 9},
    ]
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=4,
        windows_per_video=2,
        positive_paths={"positive"},
        weak_positive_paths={"positive"},
        positive_videos_per_batch=1,
        event_stage_balanced=True,
        seed=1,
    )
    batches = list(sampler)
    assert batches
    assert all(len(batch) == 4 for batch in batches)


def _geometry_checkpoint(path: Path) -> None:
    model = BBoxFallClassifier(
        temporal_model="v33-g",
        history_offsets=HISTORY_OFFSETS,
        short_history_offsets=(0, 1, 2, 3, 4, 5, 6, 7),
        long_history_offsets=(0, 4, 8, 12, 16, 20, 24, 28, 32),
        temporal_dim=16,
        visual_dim=16,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "arguments": {
                "input_height": 288,
                "input_width": 512,
                "crop_height": 128,
                "crop_width": 128,
                "crop_context": 1.25,
                "classifier_visual_dim": 16,
                "classifier_temporal_dim": 16,
                "classifier_temporal_dropout": 0.1,
                "classifier_v32_short_offsets": [0, 1, 2, 3, 4, 5, 6, 7],
                "classifier_v32_long_offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32],
                "effective_fps": 10.0,
            },
        },
        path,
    )


def test_v34_confirmer_freezes_geometry_and_predicts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=checkpoint,
        history_offsets=HISTORY_OFFSETS,
        visual_offsets=VISUAL_OFFSETS,
        global_feature_dim=12,
        roi_feature_shape=(10, 2, 2),
        temporal_dim=16,
    )
    boxes = torch.tensor([0.2, 0.1, 0.6, 0.9]).repeat(2, 15, 1)
    outputs = model(
        torch.rand(2, 8, 12),
        torch.rand(2, 8, 10, 2, 2),
        boxes,
        torch.ones(2, 15),
        torch.ones(2, 15),
    )
    assert outputs["fall_logits"].shape == (2,)
    assert outputs["global_fall_logits"].shape == (2,)
    assert outputs["roi_fall_logits"].shape == (2,)
    assert model.trainable_parameter_count < model.parameter_count
    assert all(
        not parameter.requires_grad for parameter in model.geometry.parameters()
    )


def test_confirmer_person_mil_scores_and_masks_each_person(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=checkpoint,
        history_offsets=HISTORY_OFFSETS,
        visual_offsets=VISUAL_OFFSETS,
        global_feature_dim=12,
        roi_feature_shape=(10, 2, 2),
        temporal_dim=16,
    )
    boxes = torch.tensor([0.2, 0.1, 0.6, 0.9]).repeat(2, 15, 2, 1)
    actor_valid = torch.ones(2, 15, 2)
    actor_valid[1, :, 1] = 0.0
    outputs = model(
        torch.rand(2, 8, 12),
        torch.rand(2, 8, 2, 10, 2, 2),
        boxes,
        torch.ones(2, 15),
        actor_valid,
    )
    assert outputs["person_fall_logits"].shape == (2, 2)
    assert outputs["person_falling_logits"].shape == (2, 2)
    assert outputs["person_fallen_logits"].shape == (2, 2)
    assert outputs["selected_person_index"].shape == (2,)
    torch.testing.assert_close(
        outputs["action_logits"], outputs["person_action_logits"][:, 0]
    )
    assert outputs["person_valid_mask"].tolist() == [
        [True, True],
        [True, False],
    ]
    torch.testing.assert_close(
        outputs["fall_logits"],
        outputs["person_fall_logits"].masked_fill(
            ~outputs["person_valid_mask"],
            torch.finfo(outputs["fall_logits"].dtype).min,
        ).max(dim=1).values,
    )

    model.person_mil_temperature = 0.25
    aggregated = model._aggregate_people(
        torch.tensor([[2.0, 2.0], [2.0, -5.0]]),
        torch.tensor([[True, True], [True, False]]),
    )
    torch.testing.assert_close(aggregated, torch.tensor([2.0, 2.0]))


def test_v35_confirmer_can_jointly_train_geometry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=checkpoint,
        history_offsets=HISTORY_OFFSETS,
        visual_offsets=VISUAL_OFFSETS,
        global_feature_dim=12,
        roi_feature_shape=(10, 2, 2),
        temporal_dim=16,
        freeze_geometry=False,
    )
    model.train()
    boxes = torch.tensor([0.2, 0.1, 0.6, 0.9]).repeat(2, 15, 1)
    outputs = model(
        torch.rand(2, 8, 12),
        torch.rand(2, 8, 10, 2, 2),
        boxes,
        torch.ones(2, 15),
        torch.ones(2, 15),
    )
    outputs["fall_logits"].sum().backward()
    assert model.geometry.training
    assert model.trainable_parameter_count == model.parameter_count
    assert any(
        parameter.grad is not None for parameter in model.geometry.parameters()
    )

    optimizer = _optimizer(
        SimpleNamespace(
            train_geometry=True,
            learning_rate=6e-4,
            geometry_lr_factor=0.1,
            weight_decay=1e-4,
        ),
        model,
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"confirmer", "geometry"}
    assert groups["confirmer"]["lr"] == 6e-4
    assert abs(groups["geometry"]["lr"] - 6e-5) < 1e-12


def test_v36_cross_attention_requires_and_uses_visual_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=checkpoint,
        history_offsets=HISTORY_OFFSETS,
        visual_offsets=VISUAL_OFFSETS,
        global_feature_dim=12,
        roi_feature_shape=(10, 2, 2),
        temporal_dim=16,
        fusion_mode=FYGR_FUSION_CROSS_ATTENTION,
        attention_heads=4,
        attention_ffn_dim=32,
    )
    boxes = torch.tensor([0.2, 0.1, 0.6, 0.9]).repeat(2, 15, 1)
    global_features = torch.rand(2, 8, 12, requires_grad=True)
    roi_features = torch.rand(2, 8, 10, 2, 2, requires_grad=True)
    valid_mask = torch.ones(2, 15)
    actor_valid_mask = torch.ones(2, 15)
    actor_valid_mask[1] = 0.0
    outputs = model(
        global_features,
        roi_features,
        boxes,
        valid_mask,
        actor_valid_mask,
    )
    weights = outputs["fusion_attention_weights"]
    assert weights.shape == (2, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(weights[1], torch.tensor([1.0, 0.0]))
    outputs["fall_logits"].sum().backward()
    assert global_features.grad is not None
    assert bool(torch.isfinite(global_features.grad).all())
    assert roi_features.grad is not None
    assert float(roi_features.grad[0].abs().sum()) > 0.0
    assert model.fusion is None

    with pytest.raises(ValueError, match="at least one valid visual token"):
        model(
            global_features.detach(),
            roi_features.detach(),
            boxes,
            torch.zeros_like(valid_mask),
            torch.zeros_like(actor_valid_mask),
        )


def test_v36_training_arguments_validate_attention_shape() -> None:
    args = build_parser().parse_args(
        [
            "--fusion-mode",
            FYGR_FUSION_CROSS_ATTENTION,
            "--temporal-dim",
            "18",
            "--attention-heads",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="divisible"):
        _validate(args)


def test_v38_temporal_tokens_use_causal_visual_values(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=checkpoint,
        history_offsets=HISTORY_OFFSETS,
        visual_offsets=VISUAL_OFFSETS,
        global_feature_dim=12,
        roi_feature_shape=(10, 2, 2),
        temporal_dim=16,
        fusion_mode=FYGR_FUSION_TEMPORAL_TOKENS,
        attention_heads=4,
        attention_ffn_dim=32,
    )
    boxes = torch.tensor([0.2, 0.1, 0.6, 0.9]).repeat(2, 15, 1)
    global_features = torch.rand(2, 8, 12, requires_grad=True)
    roi_features = torch.rand(2, 8, 10, 2, 2, requires_grad=True)
    outputs = model(
        global_features,
        roi_features,
        boxes,
        torch.ones(2, 15),
        torch.ones(2, 15),
    )
    weights = outputs["fusion_attention_weights"]
    assert weights.shape == (2, 15, 16)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(2, 15), atol=1e-6, rtol=1e-6
    )
    # The oldest geometry query (offset 32) can only retrieve offset-32
    # Global/ROI values, stored at token positions 0 and 8.
    disallowed = [index for index in range(16) if index not in (0, 8)]
    assert torch.count_nonzero(weights[:, 0, disallowed]) == 0
    assert outputs["fusion_state"].shape == (2, 16)
    assert outputs["fusion_pre_norm_state"].shape == (2, 16)
    outputs["fall_logits"].sum().backward()
    assert global_features.grad is not None
    assert float(global_features.grad.abs().sum()) > 0.0
    assert roi_features.grad is not None
    assert float(roi_features.grad.abs().sum()) > 0.0


def test_v38_temporal_tokens_handle_missing_history(tmp_path: Path) -> None:
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=checkpoint,
        history_offsets=HISTORY_OFFSETS,
        visual_offsets=VISUAL_OFFSETS,
        global_feature_dim=12,
        roi_feature_shape=(10, 2, 2),
        temporal_dim=16,
        fusion_mode=FYGR_FUSION_TEMPORAL_TOKENS,
        attention_heads=4,
        attention_ffn_dim=32,
    )
    valid = torch.zeros(1, 15)
    valid[:, :4] = 1.0
    boxes = torch.tensor([0.2, 0.1, 0.6, 0.9]).repeat(1, 15, 1)
    outputs = model(
        torch.rand(1, 8, 12),
        torch.rand(1, 8, 10, 2, 2),
        boxes,
        valid,
        valid,
    )
    assert bool(torch.isfinite(outputs["fall_logits"]).all())
    assert outputs["fusion_query_valid_mask"].sum() == 3
    assert outputs["fusion_key_valid_mask"].sum() == 4

    no_actor = model(
        torch.rand(1, 8, 12),
        torch.rand(1, 8, 10, 2, 2),
        boxes,
        torch.ones(1, 15),
        torch.zeros(1, 15),
    )
    assert bool(torch.isfinite(no_actor["fall_logits"]).all())
    assert bool(no_actor["fusion_query_valid_mask"][0, -1])
    assert float(
        no_actor["fusion_attention_weights"][0, -1].sum().detach()
    ) == pytest.approx(1.0)


def test_fusion_hard_adl_ranking_is_domain_and_category_balanced() -> None:
    base = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            event_state=0.0,
            onset=0.0,
            action=0.0,
            control=0.0,
            consistency=0.0,
            video_mil=0.0,
            video_ranking=0.0,
        )
    )
    criterion = FYGRCriterion(
        base,
        global_aux_weight=0.0,
        roi_aux_weight=0.0,
        weak_video_aux_weight=0.0,
        fusion_hard_adl_ranking_weight=1.0,
        fusion_hard_adl_ranking_margin=0.2,
        fusion_hard_adl_categories=("lie_down", "sit"),
        hard_negative_anchor_categories={
            ("omni_lie", 0): "lie_down",
            ("omni_sit", 0): "sit",
            ("fvd_lie", 0): "lie_down",
        },
    )
    logits = torch.tensor(
        [0.2, 0.4, 0.8, 0.7, 0.3, 0.9], requires_grad=True
    )
    outputs = {
        "stage_logits": torch.zeros(6, 3),
        "fall_logits": logits,
        "onset_logits": torch.zeros(6),
        "action_logits": torch.zeros(6, 16),
    }
    targets = [
        {
            "stage_label": torch.tensor(0),
            "action_label": torch.tensor(9),
            "event_target": torch.tensor(0.0),
            "onset_target": torch.tensor(0.0),
            "event_loss_weight": torch.tensor(0.0),
            "action_loss_weight": torch.tensor(0.0),
            "control_loss_weight": torch.tensor(0.0),
        }
        for _ in logits
    ]
    metadata = [
        {
            "domain": "omnifall",
            "path": "omni_fall",
            "window_index": index,
            "video_has_fall_event": True,
        }
        for index in (0, 1)
    ] + [
        {
            "domain": "omnifall",
            "path": "omni_lie",
            "window_index": 0,
            "video_has_fall_event": False,
        },
        {
            "domain": "omnifall",
            "path": "omni_sit",
            "window_index": 0,
            "video_has_fall_event": False,
        },
        {
            "domain": "fall_video_dataset",
            "path": "fvd_fall",
            "window_index": 0,
            "video_has_fall_event": True,
        },
        {
            "domain": "fall_video_dataset",
            "path": "fvd_lie",
            "window_index": 0,
            "video_has_fall_event": False,
        },
    ]
    losses = criterion(outputs, targets, metadata)
    assert float(losses["loss_fusion_hard_adl_ranking"].detach()) > 0.0
    assert losses["num_fusion_ranking_domains"].item() == 2
    assert losses["num_fusion_ranking_categories"].item() == 3
    losses["loss"].backward()
    assert logits.grad is not None
    assert float(logits.grad[1]) < 0.0
    assert float(logits.grad[2]) > 0.0
    assert float(logits.grad[3]) > 0.0
    assert float(logits.grad[4]) < 0.0
    assert float(logits.grad[5]) > 0.0


def test_fusion_hard_tail_ranking_uses_chunk_topk_and_low_positive_tail() -> None:
    base = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            event_state=0.0,
            onset=0.0,
            action=0.0,
            control=0.0,
            consistency=0.0,
            video_mil=0.0,
            video_ranking=0.0,
        )
    )
    criterion = FYGRCriterion(
        base,
        global_aux_weight=0.0,
        roi_aux_weight=0.0,
        weak_video_aux_weight=0.0,
        fusion_hard_tail_ranking_weight=1.0,
        fusion_hard_tail_ranking_margin=0.2,
        fusion_hard_tail_negative_topk=2,
        fusion_hard_tail_positive_quantile=0.5,
        fusion_hard_tail_categories=("lie_down",),
        hard_negative_window_categories={
            ("hard_lie", 0): ("lie_down",),
            ("hard_lie", 1): ("lie_down",),
            ("hard_lie", 2): ("lie_down",),
        },
    )
    logits = torch.tensor(
        [0.1, 0.3, 1.0, 0.9, -1.0], requires_grad=True
    )
    outputs = {
        "stage_logits": torch.zeros(5, 3),
        "fall_logits": logits,
        "onset_logits": torch.zeros(5),
        "action_logits": torch.zeros(5, 16),
    }
    targets = [
        {
            "stage_label": torch.tensor(0),
            "action_label": torch.tensor(9),
            "event_target": torch.tensor(0.0),
            "onset_target": torch.tensor(0.0),
            "event_loss_weight": torch.tensor(0.0),
            "action_loss_weight": torch.tensor(0.0),
            "control_loss_weight": torch.tensor(0.0),
        }
        for _ in logits
    ]
    metadata = [
        {
            "domain": "omnifall",
            "path": "fall",
            "window_index": index,
            "video_has_fall_event": True,
            "weak_video_label": False,
            "event_target": True,
        }
        for index in (0, 1)
    ] + [
        {
            "domain": "omnifall",
            "path": "hard_lie",
            "window_index": index,
            "video_has_fall_event": False,
            "weak_video_label": False,
            "event_target": False,
        }
        for index in (0, 1, 2)
    ]

    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()

    assert losses["num_fusion_hard_tail_domains"].item() == 1
    assert losses["num_fusion_hard_tail_categories"].item() == 1
    assert losses["num_fusion_hard_tail_positive_videos"].item() == 1
    assert losses["num_fusion_hard_tail_negative_windows"].item() == 3
    assert logits.grad is not None
    assert float(logits.grad[0]) < 0.0
    assert float(logits.grad[1]) == 0.0
    assert float(logits.grad[2]) > 0.0
    assert float(logits.grad[3]) > 0.0
    assert float(logits.grad[4]) == 0.0


def test_fusion_hard_tail_ranking_includes_weak_unknown_domain() -> None:
    base = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            event_state=0.0,
            onset=0.0,
            action=0.0,
            control=0.0,
            consistency=0.0,
            video_mil=0.0,
            video_ranking=0.0,
        )
    )
    criterion = FYGRCriterion(
        base,
        global_aux_weight=0.0,
        roi_aux_weight=0.0,
        weak_video_aux_weight=0.0,
        weak_video_topk=2,
        fusion_hard_tail_ranking_weight=1.0,
        fusion_hard_tail_categories=("score_mined_unknown",),
        hard_negative_window_categories={
            ("fvd_negative", 0): ("score_mined_unknown",),
            ("fvd_negative", 1): ("score_mined_unknown",),
        },
    )
    logits = torch.tensor([0.0, 0.2, 0.9, 0.8], requires_grad=True)
    outputs = {
        "stage_logits": torch.zeros(4, 3),
        "fall_logits": logits,
        "onset_logits": torch.zeros(4),
        "action_logits": torch.zeros(4, 16),
    }
    targets = [
        {
            "stage_label": torch.tensor(0),
            "action_label": torch.tensor(9),
            "event_target": torch.tensor(0.0),
            "onset_target": torch.tensor(0.0),
            "event_loss_weight": torch.tensor(0.0),
            "action_loss_weight": torch.tensor(0.0),
            "control_loss_weight": torch.tensor(0.0),
        }
        for _ in logits
    ]
    metadata = [
        {
            "domain": "fall_video_dataset",
            "path": "fvd_positive",
            "window_index": index,
            "video_has_fall_event": True,
            "weak_video_label": True,
            "event_target": False,
        }
        for index in (0, 1)
    ] + [
        {
            "domain": "fall_video_dataset",
            "path": "fvd_negative",
            "window_index": index,
            "video_has_fall_event": False,
            "weak_video_label": True,
            "event_target": False,
        }
        for index in (0, 1)
    ]

    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()

    assert losses["num_fusion_hard_tail_domains"].item() == 1
    assert losses["num_fusion_hard_tail_unknown_windows"].item() == 2
    assert logits.grad is not None
    assert bool(torch.all(logits.grad[:2] < 0))
    assert bool(torch.all(logits.grad[2:] > 0))


def test_fusion_stability_accumulator_reports_token_statistics() -> None:
    meter = _FusionStabilityAccumulator(
        use_global=True, use_roi=True, visual_slots=2
    )
    meter.update(
        {
            "fusion_state": torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
            "fusion_attention_weights": torch.tensor(
                [
                    [[0.5, 0.0, 0.25, 0.25]],
                    [[0.25, 0.25, 0.25, 0.25]],
                ]
            ),
            "fusion_key_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        }
    )
    result = meter.result()
    assert result["fusion_state_norm_mean"] == 3.5
    assert result["fusion_global_mass_mean"] == 0.5
    assert result["fusion_roi_mass_mean"] == 0.5
    assert 0.0 < result["fusion_attention_entropy_mean"] <= 1.0


def test_hard_tail_metric_accumulator_reports_domain_and_action_tails() -> None:
    meter = _HardTailMetricAccumulator()
    meter.update(
        {"fall_logits": torch.tensor([2.0, 1.0, -1.0, 0.5])},
        [
            {
                "domain": "omnifall",
                "path": "positive",
                "video_has_fall_event": True,
                "window_action_label": 1,
            },
            {
                "domain": "omnifall",
                "path": "negative_lie",
                "video_has_fall_event": False,
                "window_action_label": 5,
            },
            {
                "domain": "omnifall",
                "path": "negative_sit",
                "video_has_fall_event": False,
                "window_action_label": 3,
            },
            {
                "domain": "fall_video_dataset",
                "path": "negative_unknown",
                "video_has_fall_event": False,
                "window_action_label": 9,
            },
        ],
    )

    result = meter.result()

    assert result["omnifall_tail_positive_count"] == 1.0
    assert result["omnifall_tail_negative_count"] == 2.0
    assert result["omnifall_tail_negative_lie_down_count"] == 1.0
    assert result["omnifall_tail_negative_sit_down_count"] == 1.0
    assert result["fall_video_dataset_tail_negative_other_count"] == 1.0


def test_v38_model_version_preserves_multidomain_and_person_contract() -> None:
    args = build_parser().parse_args(
        ["--fusion-mode", FYGR_FUSION_TEMPORAL_TOKENS]
    )
    assert _model_version(
        args, multidomain=True, person_mil=False
    ) == "v3.8-md-temporal-token"
    assert _model_version(
        args, multidomain=True, person_mil=True
    ) == "v3.8-md-person-mil-temporal-token"


def test_v39_model_version_requires_summary_ranking_contract() -> None:
    args = build_parser().parse_args(
        [
            "--fusion-mode",
            FYGR_FUSION_CROSS_ATTENTION,
            "--fusion-hard-adl-ranking-loss-weight",
            "0.25",
        ]
    )
    assert _model_version(
        args, multidomain=True, person_mil=False
    ) == "v3.9-md-summary-ranking"
    assert _model_version(
        args, multidomain=False, person_mil=False
    ) == "v3.9-summary-ranking"

    args.fusion_hard_adl_ranking_loss_weight = 0.0
    assert _model_version(
        args, multidomain=True, person_mil=False
    ) == "v3.6-md-gcva"


def test_v310_model_version_requires_hard_tail_ranking_contract() -> None:
    args = build_parser().parse_args(
        [
            "--fusion-mode",
            FYGR_FUSION_CROSS_ATTENTION,
            "--fusion-hard-tail-ranking-loss-weight",
            "0.25",
        ]
    )
    assert _model_version(
        args, multidomain=True, person_mil=False
    ) == "v3.10-md-hard-tail-ranking"
    assert _model_version(
        args, multidomain=False, person_mil=False
    ) == "v3.10-hard-tail-ranking"


def test_manifest_model_selection_uses_end_to_end_metric() -> None:
    manifest = Path("split_manifest.csv")
    assert _selection_metric_name(
        multidomain=False, split_manifest=manifest
    ) == "end_to_end_video_mean_precision"
    assert _selection_metric_name(
        multidomain=True, split_manifest=manifest
    ) == "macro_end_to_end_video_mean_precision"


def test_manifest_model_selection_requires_complete_validation() -> None:
    args = build_parser().parse_args(
        ["--split-manifest", "split_manifest.csv", "--max-val-batches", "1"]
    )
    with pytest.raises(ValueError, match="complete validation"):
        _validate(args)


def test_train_overlay_cache_does_not_require_a_validation_pair() -> None:
    args = build_parser().parse_args(
        ["--train-overlay-cache", "corrected-train-cache"]
    )

    _validate(args)
    assert args.train_overlay_cache == [Path("corrected-train-cache")]


def test_candidate_mining_selects_train_windows(tmp_path: Path) -> None:
    cache_root = tmp_path / "train_cache"
    writer = FrozenFeatureCacheWriter(
        cache_root,
        metadata={
            "split": "train",
            "history_offsets": list(HISTORY_OFFSETS),
            "visual_offsets": list(VISUAL_OFFSETS),
        },
    )
    writer.write_video(
        "video/sample",
        frame_indices=np.asarray(HISTORY_OFFSETS),
        visual_frame_indices=np.asarray(VISUAL_OFFSETS),
        actor_boxes=np.tile(
            np.asarray([0.2, 0.1, 0.6, 0.9], dtype=np.float32),
            (15, 1),
        ),
        actor_valid_mask=np.ones(15, dtype=np.float32),
        visual_actor_boxes=np.ones((8, 4), dtype=np.float32),
        global_features=np.ones((8, 5), dtype=np.float16),
        roi_features=np.ones((8, 4, 2, 2), dtype=np.float16),
        source_size=(720, 1280),
        records=[_cache_record(0), _cache_record(1)],
    )
    writer.close()
    checkpoint = tmp_path / "geometry.pt"
    _geometry_checkpoint(checkpoint)
    output = tmp_path / "candidates.jsonl"
    run_mining(
        argparse.Namespace(
            cache=cache_root,
            geometry_checkpoint=checkpoint,
            output=output,
            geometry_threshold=0.0,
            hard_action_ids=(3, 5, 10, 12),
            easy_negative_keep=0.0,
            batch_size=2,
            workers=0,
            device="cpu",
            seed=1,
            log_interval=1,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["selected"] for row in rows)
