from __future__ import annotations

from pathlib import Path

import torch
import yaml
import pytest

from edgefall_st.training.sampler import VideoGroupedBatchSampler
from edgefall_st.training.train import (
    _ClassifierMetricAccumulator,
    _augment_bbox_classifier_inputs,
    _classifier_event_stage_balanced,
    _classifier_history_offsets,
    _close_logger,
    _create_run_artifacts,
    _make_model_and_criterion,
    _make_optimizer,
    _truncate_history_mask,
    _validate_arguments,
    build_parser,
)


def _windows() -> list[dict[str, object]]:
    return [
        {
            "path": f"fall/video_{video_index}",
            "end_time_s": window_index * 0.4,
        }
        for video_index in range(3)
        for window_index in range(4)
    ]


def test_video_grouped_sampler_combines_adjacent_windows() -> None:
    windows = _windows()
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=6,
        windows_per_video=2,
        shuffle=False,
        drop_last=True,
    )

    batches = list(sampler)

    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 6
        paths = [str(windows[index]["path"]) for index in batch]
        assert {path: paths.count(path) for path in set(paths)} == {
            "fall/video_0": 2,
            "fall/video_1": 2,
            "fall/video_2": 2,
        }
        for path in set(paths):
            times = [
                float(windows[index]["end_time_s"])
                for index in batch
                if windows[index]["path"] == path
            ]
            assert times[1] - times[0] == pytest.approx(0.4)


def test_sampler_epoch_shuffle_is_reproducible() -> None:
    first = VideoGroupedBatchSampler(
        _windows(),
        batch_size=6,
        windows_per_video=2,
        seed=17,
    )
    second = VideoGroupedBatchSampler(
        _windows(),
        batch_size=6,
        windows_per_video=2,
        seed=17,
    )

    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)


def test_sampler_keeps_video_cohorts_in_consecutive_batches() -> None:
    windows = [
        {
            "path": f"fall/video_{video_index}",
            "end_time_s": window_index * 0.4,
        }
        for video_index in range(4)
        for window_index in range(6)
    ]
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=4,
        windows_per_video=2,
        shuffle=False,
    )

    batches = list(sampler)
    first_cohort_paths = [
        {str(windows[index]["path"]) for index in batch}
        for batch in batches[:3]
    ]

    assert first_cohort_paths == [
        {"fall/video_0", "fall/video_1"},
        {"fall/video_0", "fall/video_1"},
        {"fall/video_0", "fall/video_1"},
    ]


def test_sampler_rotates_dropped_video_boundary_windows_across_epochs() -> None:
    windows = [
        {
            "path": f"fall/video_{video_index}",
            "end_time_s": window_index * 0.4,
        }
        for video_index in range(2)
        for window_index in range(13)
    ]
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=24,
        windows_per_video=12,
        shuffle=True,
        drop_last=True,
        seed=7,
    )
    selected_for_first_video: set[int] = set()

    for epoch in range(8):
        sampler.set_epoch(epoch)
        selected_for_first_video.update(
            index
            for batch in sampler
            for index in batch
            if str(windows[index]["path"]) == "fall/video_0"
        )

    assert selected_for_first_video == set(range(13))


def test_event_balanced_sampler_puts_a_fall_video_in_every_batch() -> None:
    windows = [
        {
            "path": f"video_{video_index}",
            "end_time_s": window_index * 0.4,
            "action_label": (
                1
                if video_index < 2 and window_index in (2, 3)
                else 5
            ),
        }
        for video_index in range(6)
        for window_index in range(4)
    ]
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=6,
        windows_per_video=2,
        shuffle=False,
        positive_paths={"video_0", "video_1"},
        positive_videos_per_batch=1,
    )

    batches = list(sampler)

    assert len(batches) == 4
    for batch in batches:
        paths = {str(windows[index]["path"]) for index in batch}
        positive_paths = paths & {"video_0", "video_1"}
        assert len(positive_paths) == 1
        assert any(
            int(windows[index]["action_label"]) == 1
            for index in batch
            if windows[index]["path"] in positive_paths
        )


def test_event_stage_balancing_rotates_falling_and_fallen_chunks() -> None:
    windows = [
        {
            "path": "fall/positive",
            "end_time_s": window_index * 0.4,
            "action_label": (1 if window_index in (1, 2) else 2),
        }
        for window_index in range(8)
    ]
    windows.extend(
        {
            "path": f"normal/video_{video_index}",
            "end_time_s": window_index * 0.4,
            "action_label": 5,
        }
        for video_index in range(4)
        for window_index in range(8)
    )
    sampler = VideoGroupedBatchSampler(
        windows,
        batch_size=6,
        windows_per_video=2,
        shuffle=False,
        positive_paths={"fall/positive"},
        positive_videos_per_batch=1,
        event_stage_balanced=True,
    )

    sampled_positive_chunks = [
        tuple(
            index
            for index in batch
            if windows[index]["path"] == "fall/positive"
        )
        for batch in sampler
    ]
    sampled_actions = {
        int(windows[index]["action_label"])
        for chunk in sampled_positive_chunks
        for index in chunk
    }

    assert sampled_actions == {1, 2}
    assert len(set(sampled_positive_chunks)) > 1


def test_classifier_video_metrics_use_event_table_target() -> None:
    accumulator = _ClassifierMetricAccumulator()
    outputs = {
        "stage_logits": torch.tensor(
            [[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]],
        ),
        "fall_logits": torch.tensor([2.0, -2.0]),
        "onset_logits": torch.zeros(2),
    }
    targets = [
        {
            "stage_label": torch.tensor(2),
            "onset_target": torch.tensor(0.0),
            "action_loss_weight": torch.tensor(1.0),
        },
        {
            "stage_label": torch.tensor(0),
            "onset_target": torch.tensor(0.0),
            "action_loss_weight": torch.tensor(1.0),
        },
    ]
    metadata = [
        {
            "path": "fallen/negative",
            "video_has_fall_event": False,
        },
        {
            "path": "fall/positive",
            "video_has_fall_event": True,
        },
    ]

    accumulator.update(outputs, targets, metadata)
    metrics = accumulator.result()

    assert accumulator.positive_videos == {"fall/positive"}
    assert metrics["video_average_precision"] == pytest.approx(0.5)
    assert metrics["video_mean_precision"] == pytest.approx(0.5)


def test_history_truncation_preserves_current_and_existing_padding() -> None:
    torch.manual_seed(7)
    valid_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
    )

    truncated = _truncate_history_mask(valid_mask, probability=1.0)

    assert torch.all(truncated[:, 0] == 1.0)
    assert torch.all(truncated <= valid_mask)
    assert torch.all(truncated.sum(dim=1) < valid_mask.sum(dim=1))


def test_bbox_augmentation_flips_boxes_and_converts_to_grayscale() -> None:
    images = torch.zeros(2, 3, 4, 6)
    images[:, 0] = 1.0
    boxes = torch.tensor(
        [[0.1, 0.2, 0.4, 0.8], [0.0, 0.0, 0.0, 0.0]],
    )
    valid = torch.tensor([1.0, 0.0])

    augmented_images, augmented_boxes = (
        _augment_bbox_classifier_inputs(
            images,
            boxes,
            valid,
            horizontal_flip_prob=1.0,
            grayscale_prob=1.0,
            photometric_jitter=0.0,
            bbox_jitter=0.0,
        )
    )

    torch.testing.assert_close(
        augmented_boxes[0],
        torch.tensor([0.6, 0.2, 0.9, 0.8]),
    )
    torch.testing.assert_close(augmented_boxes[1], boxes[1])
    torch.testing.assert_close(
        augmented_images[:, 0],
        augmented_images[:, 1],
    )
    torch.testing.assert_close(
        augmented_images[:, 1],
        augmented_images[:, 2],
    )


def test_geometry_augmentation_does_not_touch_rgb_tensor() -> None:
    images = torch.randint(0, 256, (2, 3, 4, 6), dtype=torch.uint8)
    boxes = torch.tensor(
        [[0.1, 0.2, 0.4, 0.8], [0.0, 0.0, 0.0, 0.0]],
    )
    valid = torch.tensor([1.0, 0.0])

    augmented_images, augmented_boxes = _augment_bbox_classifier_inputs(
        images,
        boxes,
        valid,
        horizontal_flip_prob=1.0,
        grayscale_prob=1.0,
        photometric_jitter=1.0,
        bbox_jitter=0.0,
        augment_images=False,
    )

    assert augmented_images.data_ptr() == images.data_ptr()
    torch.testing.assert_close(augmented_images, images)
    torch.testing.assert_close(
        augmented_boxes[0],
        torch.tensor([0.6, 0.2, 0.9, 0.8]),
    )


def test_argparse_parameters_create_two_log_files(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--windows-per-video",
            "2",
            "--workers",
            "0",
            "--warmup-epochs",
            "0",
            "--freeze-backbone-epochs",
            "0",
            "--log-root",
            str(tmp_path / "log"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
        ]
    )
    _validate_arguments(args)

    artifacts = _create_run_artifacts(args, device=torch.device("cpu"))
    artifacts.logger.info("test log entry")
    _close_logger(artifacts.logger)

    assert sorted(path.name for path in artifacts.log_dir.iterdir()) == [
        "parameters.yaml",
        "train.log",
    ]
    parameters = yaml.safe_load(
        artifacts.parameters_path.read_text(encoding="utf-8")
    )
    assert parameters["arguments"]["batch_size"] == 4
    assert parameters["arguments"]["workers"] == 0
    assert parameters["arguments"]["video_frame_cache_size"] == 4
    assert parameters["arguments"]["uint8_images"] is True
    assert parameters["arguments"]["deduplicate_backbone_frames"] is True
    assert parameters["arguments"]["amp_dtype"] == "bfloat16"
    assert parameters["resolved"]["device"] == "cpu"
    assert "test log entry" in artifacts.log_path.read_text(encoding="utf-8")
    assert artifacts.checkpoint_dir.parent == tmp_path / "checkpoints"


def test_one_epoch_pilot_accepts_longer_freeze_and_warmup_defaults() -> None:
    args = build_parser().parse_args(["--epochs", "1"])

    _validate_arguments(args)

    assert args.freeze_backbone_epochs == 2
    assert args.warmup_epochs == 3


def test_bbox_classifier_mode_has_semantic_only_defaults() -> None:
    args = build_parser().parse_args(
        ["--training-mode", "bbox-classifier"]
    )

    _validate_arguments(args)

    assert args.classifier_fall_loss_weight == 2.0
    assert args.classifier_action_loss_weight == 0.05
    assert args.classifier_consistency_loss_weight == 0.25
    assert args.classifier_video_mil_loss_weight == 0.5
    assert args.classifier_video_ranking_loss_weight == 0.25
    assert args.classifier_video_ranking_margin == 0.2
    assert args.positive_videos_per_batch == 1
    assert args.history_truncation_prob == 0.2
    assert args.horizontal_flip_prob == 0.5
    assert args.grayscale_prob == 0.2
    assert args.photometric_jitter == 0.15
    assert args.bbox_jitter == 0.05
    assert args.train_actor_crops_root is None
    assert args.val_actor_crops_root is None
    assert args.classifier_temporal_model == "dual-tcn"
    assert args.classifier_short_horizon == 4
    assert args.classifier_spatial_pool_size is None
    assert args.classifier_visual_input_size == 64
    assert args.classifier_geometry_init_checkpoint is None
    assert args.classifier_freeze_geometry is False
    assert args.classifier_control_loss_weight is None
    assert args.classifier_video_pooling is None
    assert args.crop_height == 128
    assert args.crop_width == 128

    model, criterion = _make_model_and_criterion(args)
    assert model.temporal_model == "dual-tcn"
    assert model.spatial_pool_size == 1
    assert criterion.loss_weights.control == 0.0
    assert criterion.video_pooling == "max"
    assert criterion.loss_weights.consistency == 0.25


def test_v32_defaults_enable_regular_history_and_semantic_losses() -> None:
    args = build_parser().parse_args(
        [
            "--training-mode",
            "bbox-classifier",
            "--classifier-temporal-model",
            "v32",
        ]
    )

    _validate_arguments(args)
    model, criterion = _make_model_and_criterion(args)

    assert _classifier_history_offsets(args) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
    )
    assert model.temporal_model == "v32"
    assert model.spatial_pool_size == 2
    assert criterion.loss_weights.control == 0.25
    assert criterion.video_pooling == "topk-mean"
    assert criterion.video_topk == 2


def test_v33_defaults_enable_geometry_first_training_contract() -> None:
    args = build_parser().parse_args(
        [
            "--training-mode",
            "bbox-classifier",
            "--classifier-temporal-model",
            "v33-g",
        ]
    )

    _validate_arguments(args)
    model, criterion = _make_model_and_criterion(args)

    assert model.temporal_model == "v33-g"
    assert list(model.backbone.parameters()) == []
    assert _classifier_event_stage_balanced(args) is True
    assert criterion.loss_weights.event_state == 0.5
    assert criterion.loss_weights.geometry_stage == 0.0
    assert criterion.loss_weights.geometry_fall == 0.0
    assert criterion.loss_weights.control == 0.25
    assert criterion.video_pooling == "asymmetric"
    optimizer = _make_optimizer(args, model)
    assert [group["name"] for group in optimizer.param_groups] == ["main"]

    gr_args = build_parser().parse_args(
        [
            "--training-mode",
            "bbox-classifier",
            "--classifier-temporal-model",
            "v33-gr",
        ]
    )
    gr_model, gr_criterion = _make_model_and_criterion(gr_args)
    assert gr_model.visual_input_size == 64
    assert gr_criterion.loss_weights.geometry_stage == 0.5
    assert gr_criterion.loss_weights.geometry_fall == 1.0
    gr_model.freeze_geometry_branch()
    gr_optimizer = _make_optimizer(gr_args, gr_model)
    assert [group["name"] for group in gr_optimizer.param_groups] == [
        "backbone",
        "main",
    ]
    assert all(
        parameter.requires_grad
        for group in gr_optimizer.param_groups
        for parameter in group["params"]
    )


def test_geometry_freeze_requires_v33_g_checkpoint() -> None:
    args = build_parser().parse_args(
        [
            "--training-mode",
            "bbox-classifier",
            "--classifier-temporal-model",
            "v33-gr",
            "--classifier-freeze-geometry",
        ]
    )

    with pytest.raises(ValueError, match="requires a geometry checkpoint"):
        _validate_arguments(args)


def test_bbox_classifier_old_arguments_restore_attention_model() -> None:
    args = build_parser().parse_args(
        ["--training-mode", "bbox-classifier"]
    )
    for name in (
        "classifier_temporal_model",
        "classifier_temporal_dropout",
        "classifier_short_horizon",
        "classifier_consistency_loss_weight",
        "classifier_video_mil_loss_weight",
        "classifier_video_ranking_loss_weight",
        "classifier_video_ranking_margin",
        "train_actor_crops_root",
        "val_actor_crops_root",
    ):
        delattr(args, name)

    model, criterion = _make_model_and_criterion(args)

    assert model.temporal_model == "attention"
    assert criterion.loss_weights.consistency == 0.0
    assert criterion.loss_weights.video_mil == 0.0
    assert criterion.loss_weights.video_ranking == 0.0
    assert model.pre_cropped_inputs is False
