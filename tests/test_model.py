from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from edgefall_st.models import BBoxFallClassifier, EdgeFallST
from edgefall_st.models.bbox_classifier import (
    V32_LONG_HISTORY_OFFSETS,
    V32_SHORT_HISTORY_OFFSETS,
)
from edgefall_st.runtime.postprocess import postprocess_predictions


def make_model() -> EdgeFallST:
    return EdgeFallST(
        input_size=(64, 96),
        backbone_channels=(8, 16, 24, 32, 48),
        neck_channels=32,
        embedding_dim=8,
    )


def test_frame_forward_shapes() -> None:
    model = make_model().eval()
    images = torch.randn(2, 3, 64, 96)

    with torch.inference_mode():
        outputs = model(images)

    locations = 8 * 12 + 4 * 6 + 2 * 3
    assert outputs["boxes"].shape == (2, locations, 4)
    assert outputs["objectness_logits"].shape == (2, locations)
    assert outputs["stage_logits"].shape == (2, locations, 3)
    assert outputs["onset_logits"].shape == (2, locations)
    assert outputs["track_embeddings"].shape == (2, locations, 8)
    assert outputs["action_logits"].shape == (2, locations, 16)
    assert outputs["strides"].shape == (locations,)


def test_sparse_history_forward_and_backward() -> None:
    model = make_model().train()
    slots = len(model.global_frame_offsets)
    images = torch.randn(2, slots, 3, 64, 96)
    valid = torch.ones(2, slots)
    valid[:, -2:] = 0

    outputs = model.forward_history(images, valid)
    loss = (
        outputs["objectness_logits"].mean()
        + outputs["stage_logits"].mean()
        + outputs["onset_logits"].mean()
    )
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.temporal_fusion.parameters()
    )


def test_indexed_history_matches_expanded_history_in_eval_mode() -> None:
    model = make_model().eval()
    slots = len(model.global_frame_offsets)
    unique_images = torch.randn(5, 3, 64, 96)
    history_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, -1],
            [2, 3, 4, 0, 1, 2, 3, -1],
        ],
        dtype=torch.long,
    )
    assert history_indices.shape[1] == slots
    valid_mask = history_indices.ge(0).to(torch.float32)
    expanded = unique_images[history_indices.clamp_min(0)].clone()
    expanded[history_indices < 0] = 0

    with torch.inference_mode():
        expected = model.forward_history(expanded, valid_mask)
        actual = model.forward_indexed_history(
            unique_images,
            history_indices,
            valid_mask,
        )

    for name in (
        "boxes",
        "objectness_logits",
        "stage_logits",
        "onset_logits",
        "track_embeddings",
        "action_logits",
    ):
        torch.testing.assert_close(actual[name], expected[name])


def test_indexed_history_backpropagates_to_unique_frames() -> None:
    model = make_model().train()
    unique_images = torch.randn(6, 3, 64, 96, requires_grad=True)
    history_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0, 1],
            [2, 3, 4, 5, 0, 1, 2, 3],
        ],
        dtype=torch.long,
    )

    outputs = model.forward_indexed_history(unique_images, history_indices)
    loss = (
        outputs["objectness_logits"].mean()
        + outputs["stage_logits"].mean()
        + outputs["onset_logits"].mean()
    )
    loss.backward()

    assert unique_images.grad is not None
    assert torch.count_nonzero(unique_images.grad) > 0


def test_stream_state_accumulates_and_resets() -> None:
    model = make_model().eval()
    state = model.create_stream_state()
    image = torch.randn(1, 3, 64, 96)

    with torch.inference_mode():
        first = model.forward_stream(image, state)
        second = model.forward_stream(image, state)

    assert len(state.features["p3"]) == 2
    assert first["boxes"].shape == second["boxes"].shape
    state.reset()
    assert all(len(values) == 0 for values in state.features.values())


def test_default_model_stays_below_parameter_limit() -> None:
    model = EdgeFallST()

    assert model.parameter_count < 20_000_000


def _bbox_classifier() -> BBoxFallClassifier:
    return BBoxFallClassifier(
        input_size=(64, 96),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=(0, 1, 2, 4),
    )


def test_bbox_classifier_uses_boxes_without_detection_outputs() -> None:
    model = _bbox_classifier().train()
    images = torch.randn(2, 4, 3, 64, 96)
    boxes = torch.tensor(
        [
            [
                [0.1, 0.1, 0.5, 0.9],
                [0.1, 0.1, 0.5, 0.9],
                [0.1, 0.1, 0.5, 0.9],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.2, 0.4, 0.8, 0.8],
                [0.2, 0.3, 0.7, 0.8],
                [0.2, 0.2, 0.6, 0.9],
                [0.2, 0.1, 0.5, 0.9],
            ],
        ]
    )
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0] * 4])

    outputs = model.forward_history(images, boxes, valid, valid)
    loss = (
        outputs["stage_logits"].mean()
        + outputs["fall_logits"].mean()
        + outputs["onset_logits"].mean()
    )
    loss.backward()

    assert outputs["stage_logits"].shape == (2, 3)
    assert outputs["fall_logits"].shape == (2,)
    assert outputs["onset_logits"].shape == (2,)
    assert outputs["action_logits"].shape == (2, 16)
    assert "boxes" not in outputs
    assert "objectness_logits" not in outputs
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.parameters()
    )


def test_bbox_classifier_indexed_history_matches_expanded() -> None:
    model = _bbox_classifier().eval()
    unique_images = torch.randn(5, 3, 64, 96)
    unique_boxes = torch.tensor(
        [
            [0.1, 0.1, 0.5, 0.9],
            [0.2, 0.1, 0.6, 0.9],
            [0.3, 0.2, 0.8, 0.8],
            [0.2, 0.3, 0.9, 0.9],
            [0.1, 0.4, 0.9, 0.8],
        ]
    )
    indices = torch.tensor([[0, 1, 2, -1], [2, 3, 4, 0]])
    valid = indices.ge(0).float()
    expanded_images = unique_images[indices.clamp_min(0)].clone()
    expanded_boxes = unique_boxes[indices.clamp_min(0)].clone()
    actor_valid = torch.ones(5)

    with torch.inference_mode():
        expected = model.forward_history(
            expanded_images,
            expanded_boxes,
            valid,
            valid,
        )
        actual = model.forward_indexed_history(
            unique_images,
            unique_boxes,
            indices,
            valid,
            actor_valid,
        )

    for name in ("stage_logits", "fall_logits", "onset_logits", "action_logits"):
        torch.testing.assert_close(actual[name], expected[name])


def test_bbox_classifier_accepts_precomputed_actor_crops() -> None:
    model = BBoxFallClassifier(
        input_size=(64, 96),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=(0, 1, 2, 4),
        pre_cropped_inputs=True,
    ).eval()
    images = torch.randn(2, 4, 3, 32, 32)
    boxes = torch.tensor(
        [[[0.1, 0.1, 0.5, 0.9]] * 4, [[0.2, 0.2, 0.8, 0.8]] * 4],
    )
    valid = torch.ones(2, 4)

    with torch.inference_mode():
        outputs = model.forward_history(images, boxes, valid, valid)

    assert outputs["fall_logits"].shape == (2,)
    assert model.pre_cropped_inputs is True


def test_precomputed_crop_path_matches_online_roi_align() -> None:
    online = _bbox_classifier().eval()
    cached = BBoxFallClassifier(
        input_size=(64, 96),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=(0, 1, 2, 4),
        pre_cropped_inputs=True,
    ).eval()
    cached.load_state_dict(online.state_dict())
    images = torch.randn(2, 4, 3, 64, 96)
    boxes = torch.tensor(
        [[[0.1, 0.1, 0.5, 0.9]] * 4, [[0.2, 0.2, 0.8, 0.8]] * 4],
    )
    valid = torch.ones(2, 4)
    flat_crops = online.cropper(
        images.reshape(-1, 3, 64, 96),
        boxes.reshape(-1, 4),
    )
    crops = flat_crops.reshape(2, 4, 3, 32, 32)

    with torch.inference_mode():
        expected = online.forward_history(images, boxes, valid, valid)
        actual = cached.forward_history(crops, boxes, valid, valid)

    for name in ("stage_logits", "fall_logits", "onset_logits", "action_logits"):
        torch.testing.assert_close(actual[name], expected[name])


def test_bbox_classifier_masks_missing_temporal_tokens() -> None:
    model = _bbox_classifier().eval()
    visual = torch.randn(1, 4, 32)
    boxes = torch.tensor(
        [[[0.2, 0.1, 0.5, 0.9], [0.2, 0.2, 0.6, 0.9]] + [[0.0] * 4] * 2]
    )
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    altered_visual = visual.clone()
    altered_visual[:, 2:] = torch.randn_like(altered_visual[:, 2:]) * 100.0
    altered_boxes = boxes.clone()
    altered_boxes[:, 2:] = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [0.4, 0.4, 0.5, 0.5]]]
    )

    with torch.inference_mode():
        expected = model._classify(visual, boxes, valid)
        actual = model._classify(altered_visual, altered_boxes, valid)

    for name in ("stage_logits", "fall_logits", "onset_logits"):
        torch.testing.assert_close(actual[name], expected[name])


def test_bbox_classifier_keeps_legacy_attention_option() -> None:
    model = BBoxFallClassifier(
        input_size=(64, 96),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=(0, 1, 2, 4),
        temporal_model="attention",
    ).eval()
    visual = torch.randn(2, 4, 32)
    boxes = torch.rand(2, 4, 4)
    boxes[..., 2:] = torch.maximum(boxes[..., 2:], boxes[..., :2] + 0.01)
    boxes.clamp_(0.0, 1.0)
    valid = torch.ones(2, 4)

    with torch.inference_mode():
        outputs = model._classify(visual, boxes, valid)

    assert model.temporal_model == "attention"
    assert outputs["stage_logits"].shape == (2, 3)


def test_v32_classifier_preserves_spatial_layout_and_regular_timelines() -> None:
    offsets = tuple(
        sorted(set(V32_SHORT_HISTORY_OFFSETS + V32_LONG_HISTORY_OFFSETS))
    )
    model = BBoxFallClassifier(
        input_size=(64, 64),
        crop_size=(64, 64),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=offsets,
        temporal_model="v32",
        spatial_pool_size=2,
        short_history_offsets=V32_SHORT_HISTORY_OFFSETS,
        long_history_offsets=V32_LONG_HISTORY_OFFSETS,
        pre_cropped_inputs=True,
    ).train()
    images = torch.randn(2, len(offsets), 3, 64, 64)
    boxes = torch.tensor(
        [
            [[0.2, 0.1, 0.6, 0.9]] * len(offsets),
            [[0.1, 0.2, 0.8, 0.8]] * len(offsets),
        ]
    )
    valid = torch.ones(2, len(offsets))

    outputs = model.forward_history(images, boxes, valid, valid)
    loss = sum(
        outputs[name].float().mean()
        for name in (
            "stage_logits",
            "fall_logits",
            "onset_logits",
            "action_logits",
            "control_logits",
        )
    )
    loss.backward()

    assert model.spatial_pool_size == 2
    assert model.short_history_offsets == V32_SHORT_HISTORY_OFFSETS
    assert model.long_history_offsets == V32_LONG_HISTORY_OFFSETS
    assert outputs["control_logits"].shape == (2,)
    assert model.control_head is not None
    assert model.control_head.weight.grad is not None
    assert model.parameter_count < 20_000_000


def _v33_classifier(temporal_model: str) -> BBoxFallClassifier:
    offsets = tuple(
        sorted(set(V32_SHORT_HISTORY_OFFSETS + V32_LONG_HISTORY_OFFSETS))
    )
    return BBoxFallClassifier(
        input_size=(64, 64),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=offsets,
        temporal_model=temporal_model,
        spatial_pool_size=2,
        short_history_offsets=V32_SHORT_HISTORY_OFFSETS,
        long_history_offsets=V32_LONG_HISTORY_OFFSETS,
        pre_cropped_inputs=True,
    )


def test_v33_geometry_model_skips_rgb_and_predicts_independently() -> None:
    model = _v33_classifier("v33-g").eval()
    offsets = model.history_offsets
    boxes = torch.tensor(
        [
            [[0.2, 0.1, 0.5, 0.9]] * len(offsets),
            [[0.1, 0.2, 0.8, 0.8]] * len(offsets),
        ]
    )
    valid = torch.ones(2, len(offsets))
    first_images = torch.randn(2, len(offsets), 3, 32, 32)
    second_images = torch.randn_like(first_images)

    with torch.inference_mode():
        first = model.forward_history(first_images, boxes, valid, valid)
        second = model.forward_history(second_images, boxes, valid, valid)

    assert list(model.backbone.parameters()) == []
    assert first["falling_logits"].shape == (2,)
    assert first["fallen_logits"].shape == (2,)
    assert first["geometry_stage_logits"].shape == (2, 3)
    assert first["geometry_fall_logits"].shape == (2,)
    assert torch.all(first["visual_gate"] == 0)
    for name in ("stage_logits", "fall_logits", "onset_logits"):
        torch.testing.assert_close(first[name], second[name])
    torch.testing.assert_close(
        first["fall_logits"],
        first["geometry_fall_logits"],
    )


def test_v33_geometry_statistics_stay_finite_without_valid_actor_boxes() -> None:
    model = _v33_classifier("v33-g").eval()
    slots = len(model.history_offsets)
    images = torch.zeros(1, slots, 3, 32, 32)
    boxes = torch.zeros(1, slots, 4)
    valid = torch.ones(1, slots)
    actor_valid = torch.zeros(1, slots)

    with torch.inference_mode():
        outputs = model.forward_history(images, boxes, valid, actor_valid)

    for name in (
        "stage_logits",
        "fall_logits",
        "falling_logits",
        "fallen_logits",
        "geometry_stage_logits",
        "geometry_fall_logits",
    ):
        assert torch.isfinite(outputs[name]).all()


def test_v33_gated_residual_starts_from_geometry_and_can_be_dropped() -> None:
    model = _v33_classifier("v33-gr").train()
    model.visual_modality_dropout = 1.0
    slots = len(model.history_offsets)
    images = torch.randn(2, slots, 3, 32, 32)
    boxes = torch.tensor(
        [
            [[0.2, 0.1, 0.5, 0.9]] * slots,
            [[0.1, 0.2, 0.8, 0.8]] * slots,
        ]
    )
    valid = torch.ones(2, slots)

    outputs = model.forward_history(images, boxes, valid, valid)
    loss = outputs["fall_logits"].mean() + outputs["stage_logits"].mean()
    loss.backward()

    assert torch.all(outputs["visual_gate"] == 0)
    assert torch.all(outputs["visual_event_residual"] == 0)
    torch.testing.assert_close(
        outputs["stage_logits"],
        outputs["geometry_stage_logits"],
    )
    torch.testing.assert_close(
        outputs["fall_logits"],
        outputs["geometry_fall_logits"],
    )
    assert model.event_state_head is not None
    assert model.event_state_head.weight.grad is not None
    assert model.parameter_count < 20_000_000


def test_v33_gated_residual_downsamples_only_visual_backbone_input() -> None:
    model = BBoxFallClassifier(
        input_size=(64, 64),
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=tuple(
            sorted(
                set(V32_SHORT_HISTORY_OFFSETS + V32_LONG_HISTORY_OFFSETS)
            )
        ),
        temporal_model="v33-gr",
        spatial_pool_size=2,
        short_history_offsets=V32_SHORT_HISTORY_OFFSETS,
        long_history_offsets=V32_LONG_HISTORY_OFFSETS,
        pre_cropped_inputs=True,
        visual_input_size=16,
    ).eval()
    observed_shapes: list[tuple[int, ...]] = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, inputs: observed_shapes.append(tuple(inputs[0].shape))
    )
    torch.nn.init.normal_(model.visual_event_residual.weight)
    slots = len(model.history_offsets)
    images = torch.randn(1, slots, 3, 32, 32)
    boxes = torch.tensor([[[0.2, 0.1, 0.5, 0.9]] * slots])
    valid = torch.ones(1, slots)
    history_indices = torch.arange(slots).unsqueeze(0)

    with torch.inference_mode():
        outputs = model.forward_history(images, boxes, valid, valid)
        indexed_outputs = model.forward_indexed_history(
            images[0],
            boxes[0],
            history_indices,
            valid,
            torch.ones(slots),
        )
    handle.remove()

    assert model.visual_input_size == 16
    assert observed_shapes == [(8, 3, 16, 16), (8, 3, 16, 16)]
    assert model.backbone_frame_count(history_indices, 1) == 8
    assert outputs["fall_logits"].shape == (1,)
    torch.testing.assert_close(
        outputs["fall_logits"],
        indexed_outputs["fall_logits"],
    )


def test_v33_gated_residual_can_lock_geometry_checkpoint() -> None:
    geometry_model = _v33_classifier("v33-g").eval()
    residual_model = _v33_classifier("v33-gr")
    loaded = residual_model.load_geometry_state_dict(
        geometry_model.state_dict()
    )
    residual_model.freeze_geometry_branch()
    residual_model.train()
    slots = len(geometry_model.history_offsets)
    images = torch.randn(2, slots, 3, 32, 32)
    boxes = torch.tensor(
        [
            [[0.2, 0.1, 0.5, 0.9]] * slots,
            [[0.1, 0.2, 0.8, 0.8]] * slots,
        ]
    )
    valid = torch.ones(2, slots)

    with torch.inference_mode():
        expected = geometry_model.forward_history(
            images,
            boxes,
            valid,
            valid,
        )
        actual = residual_model.forward_history(
            images,
            boxes,
            valid,
            valid,
        )

    assert loaded > 0
    assert residual_model.geometry_frozen is True
    assert residual_model.backbone.training is True
    assert residual_model.geometry_summary.training is False
    assert all(
        parameter.requires_grad
        for name, parameter in residual_model.named_parameters()
        if residual_model._is_visual_parameter(name)
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in residual_model.named_parameters()
        if not residual_model._is_visual_parameter(name)
    )
    for name in ("stage_logits", "fall_logits", "onset_logits"):
        torch.testing.assert_close(actual[name], expected[name])


def test_postprocess_applies_person_threshold_and_nms() -> None:
    outputs = {
        "boxes": torch.tensor(
            [
                [
                    [0.1, 0.1, 0.5, 0.8],
                    [0.11, 0.1, 0.51, 0.8],
                    [0.7, 0.1, 0.9, 0.8],
                ]
            ],
            dtype=torch.float32,
        ),
        "objectness_logits": torch.tensor([[5.0, 4.0, -5.0]]),
        "stage_logits": torch.tensor(
            [[[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]]
        ),
        "onset_logits": torch.tensor([[2.0, 1.0, -2.0]]),
        "track_embeddings": torch.tensor(
            [[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]]
        ),
    }

    detections = postprocess_predictions(
        outputs,
        person_threshold=0.25,
        nms_iou_threshold=0.5,
    )[0]

    assert len(detections) == 1
    assert detections[0].person_score > 0.9
    assert detections[0].stage_probabilities[1] > 0.7
