from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from external_eval.fall_video_dataset.common import (
    binary_ranking_metrics,
    discover_records,
    fixed_threshold_metrics,
    normalized_group_id,
    read_pose_boxes,
    resampled_source_indices,
)
from external_eval.fall_video_dataset.evaluate_v3_yolo import (
    read_actor_box_sequence,
)
from external_eval.fall_video_dataset.evaluate_v31_yolo import (
    _state_machine_result,
)
from external_eval.fall_video_dataset.evaluate_v3 import (
    ExternalWindowPrediction,
    _load_model,
)
from edgefall_st.models import BBoxFallClassifier
from edgefall_st.models.bbox_classifier import (
    V32_LONG_HISTORY_OFFSETS,
    V32_SHORT_HISTORY_OFFSETS,
)
from edgefall_st.runtime.v31_events import V31StateMachineConfig


def _make_pair(root: Path, label: str, stem: str, csv_body: str) -> None:
    video_root = root / label / "Raw_Video"
    keypoints_root = root / label / "Keypoints_CSV"
    video_root.mkdir(parents=True, exist_ok=True)
    keypoints_root.mkdir(parents=True, exist_ok=True)
    (video_root / f"{stem}.mp4").write_bytes(b"video")
    (keypoints_root / f"{stem}_keypoints.csv").write_text(
        "Frame,Keypoint,X,Y,Confidence\n" + csv_body,
        encoding="utf-8",
    )


def test_discover_records_keeps_class_in_sample_id_and_groups_sources(
    tmp_path: Path,
) -> None:
    _make_pair(tmp_path, "Fall", "C_D_0001_resized", "")
    _make_pair(tmp_path, "No_Fall", "C_D_0001", "")

    records = discover_records(tmp_path)

    assert {record.sample_id for record in records} == {
        "Fall/C_D_0001_resized",
        "No_Fall/C_D_0001",
    }
    assert {record.group_id for record in records} == {"c_d_0001"}
    assert normalized_group_id("Example_RESIZED") == "example"


def test_read_pose_boxes_filters_confidence_and_normalizes(tmp_path: Path) -> None:
    csv_path = tmp_path / "pose.csv"
    csv_path.write_text(
        "Frame,Keypoint,X,Y,Confidence\n"
        "1,A,10,20,0.9\n"
        "1,B,30,20,0.9\n"
        "1,C,10,60,0.9\n"
        "1,D,30,60,0.9\n"
        "2,A,99,99,0.1\n",
        encoding="utf-8",
    )

    result = read_pose_boxes(
        csv_path,
        frame_width=100,
        frame_height=200,
        confidence_threshold=0.25,
        minimum_keypoints=4,
    )

    assert result.rows == 5
    assert result.observed_frames == 2
    assert result.valid_frames == 1
    assert result.boxes[1] == pytest.approx((0.1, 0.1, 0.3, 0.3))


def test_resampled_source_indices_preserve_timestamps() -> None:
    assert resampled_source_indices(10, 30.0, 10.0) == (0, 3, 6, 9)


def test_binary_metrics_match_expected_confusion_and_ranking() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, False, True, False]

    fixed = fixed_threshold_metrics(scores, labels, threshold=0.5)
    ranking = binary_ranking_metrics(scores, labels)

    assert fixed["true_positives"] == 1
    assert fixed["false_positives"] == 1
    assert fixed["true_negatives"] == 1
    assert fixed["false_negatives"] == 1
    assert ranking["average_precision"] == pytest.approx(5.0 / 6.0)


def test_read_actor_box_sequence_keeps_zero_based_frames(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text(
        '{"frame_index": 0, "actor_bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.8]}\n'
        '{"frame_index": 1, "actor_bbox_xyxy_norm": null}\n'
        '{"frame_index": 2, "actor_bbox_xyxy_norm": [-0.1, 0.3, 1.1, 0.9]}\n',
        encoding="utf-8",
    )

    result = read_actor_box_sequence(path, frame_count=3)

    assert result.observed_frames == 3
    assert result.valid_frames == 2
    assert result.frame_alignment_delta == 0
    assert result.boxes[0] == pytest.approx((0.1, 0.2, 0.4, 0.8))
    assert result.boxes[2] == pytest.approx((0.0, 0.3, 1.0, 0.9))


def test_external_v31_requires_temporally_supported_score() -> None:
    windows = [
        ExternalWindowPrediction(
            sample_index=0,
            timestamp_s=0.0,
            fall_score=0.8,
            stage_probabilities=(0.05, 0.9, 0.05),
            onset_score=0.8,
        ),
        ExternalWindowPrediction(
            sample_index=1,
            timestamp_s=0.4,
            fall_score=0.75,
            stage_probabilities=(0.05, 0.1, 0.85),
            onset_score=0.1,
        ),
    ]

    score, alarms, events = _state_machine_result(
        windows,
        V31StateMachineConfig(),
    )

    assert score == pytest.approx(0.75)
    assert alarms == 1
    assert len(events) == 1


def test_external_loader_reconstructs_v33_geometry_checkpoint(
    tmp_path: Path,
) -> None:
    offsets = tuple(
        sorted(set(V32_SHORT_HISTORY_OFFSETS + V32_LONG_HISTORY_OFFSETS))
    )
    model = BBoxFallClassifier(
        input_size=(64, 64),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=offsets,
        temporal_model="v33-g",
        short_history_offsets=V32_SHORT_HISTORY_OFFSETS,
        long_history_offsets=V32_LONG_HISTORY_OFFSETS,
        pre_cropped_inputs=True,
    )
    saved = argparse.Namespace(
        training_mode="bbox-classifier",
        classifier_temporal_model="v33-g",
        classifier_v32_short_offsets=V32_SHORT_HISTORY_OFFSETS,
        classifier_v32_long_offsets=V32_LONG_HISTORY_OFFSETS,
        classifier_spatial_pool_size=None,
        classifier_temporal_dropout=0.1,
        classifier_short_horizon=4,
        classifier_visual_modality_dropout=0.3,
        classifier_visual_gate_bias=-2.0,
        input_height=64,
        input_width=64,
        crop_height=32,
        crop_width=32,
        crop_context=1.25,
        backbone_channels=(8, 16, 24, 32, 48),
        classifier_visual_dim=32,
        classifier_temporal_dim=32,
        activation="silu",
        effective_fps=10.0,
        amp=False,
        amp_dtype="bfloat16",
    )
    checkpoint = tmp_path / "v33_g.pt"
    torch.save(
        {
            "epoch": 0,
            "arguments": vars(saved),
            "model": model.state_dict(),
        },
        checkpoint,
    )

    loaded, loaded_args, loaded_offsets, amp_enabled, _ = _load_model(
        checkpoint,
        torch.device("cpu"),
    )

    assert loaded.temporal_model == "v33-g"
    assert loaded_args.classifier_temporal_model == "v33-g"
    assert loaded_offsets == offsets
    assert list(loaded.backbone.parameters()) == []
    assert amp_enabled is False


def test_external_loader_restores_v33_visual_input_size(
    tmp_path: Path,
) -> None:
    offsets = tuple(
        sorted(set(V32_SHORT_HISTORY_OFFSETS + V32_LONG_HISTORY_OFFSETS))
    )
    model = BBoxFallClassifier(
        input_size=(64, 64),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=offsets,
        temporal_model="v33-gr",
        spatial_pool_size=2,
        short_history_offsets=V32_SHORT_HISTORY_OFFSETS,
        long_history_offsets=V32_LONG_HISTORY_OFFSETS,
        pre_cropped_inputs=True,
        visual_input_size=16,
    )
    saved = argparse.Namespace(
        training_mode="bbox-classifier",
        classifier_temporal_model="v33-gr",
        classifier_v32_short_offsets=V32_SHORT_HISTORY_OFFSETS,
        classifier_v32_long_offsets=V32_LONG_HISTORY_OFFSETS,
        classifier_spatial_pool_size=2,
        classifier_temporal_dropout=0.1,
        classifier_short_horizon=4,
        classifier_visual_modality_dropout=0.3,
        classifier_visual_gate_bias=-2.0,
        classifier_visual_input_size=16,
        input_height=64,
        input_width=64,
        crop_height=32,
        crop_width=32,
        crop_context=1.25,
        backbone_channels=(8, 16, 24, 32, 48),
        classifier_visual_dim=32,
        classifier_temporal_dim=32,
        activation="silu",
        effective_fps=10.0,
        amp=False,
        amp_dtype="bfloat16",
    )
    checkpoint = tmp_path / "v33_gr.pt"
    torch.save(
        {
            "epoch": 0,
            "arguments": vars(saved),
            "model": model.state_dict(),
        },
        checkpoint,
    )

    loaded, _, _, _, _ = _load_model(checkpoint, torch.device("cpu"))

    assert loaded.temporal_model == "v33-gr"
    assert loaded.visual_input_size == 16
