from __future__ import annotations

import math

import torch

from edgefall_st.models import BBoxFallClassifier, EdgeFallST
from edgefall_st.training import (
    AnchorFreeAssigner,
    BBoxClassifierLossWeights,
    BBoxFallCriterion,
    EdgeFallCriterion,
)


INPUT_SIZE = (64, 96)


def _strides() -> torch.Tensor:
    return torch.cat(
        [
            torch.full((8 * 12,), 8.0),
            torch.full((4 * 6,), 16.0),
            torch.full((2 * 3,), 32.0),
        ]
    )


def _target(
    *,
    boxes: list[list[float]] | None = None,
    actor_box_index: int = 1,
    actor_identity: int = 0,
) -> dict[str, torch.Tensor]:
    values = (
        [
            [0.05, 0.10, 0.35, 0.90],
            [0.50, 0.15, 0.92, 0.92],
        ]
        if boxes is None
        else boxes
    )
    count = len(values)
    return {
        "boxes": torch.tensor(values, dtype=torch.float32).reshape(-1, 4),
        "box_confidences": torch.linspace(0.9, 0.8, count),
        "source_track_ids": torch.arange(count, dtype=torch.long) + 10,
        "actor_box_index": torch.tensor(actor_box_index),
        "actor_source_track_id": torch.tensor(20),
        "actor_identity": torch.tensor(actor_identity),
        "action_label": torch.tensor(1),
        "stage_label": torch.tensor(1),
        "onset_target": torch.tensor(1.0),
        "source_onset_target": torch.tensor(0.0),
        "action_loss_weight": torch.tensor(1.0),
        "track_loss_weight": torch.tensor(1.0),
    }


def _small_model() -> EdgeFallST:
    return EdgeFallST(
        input_size=INPUT_SIZE,
        backbone_channels=(8, 16, 24, 32, 48),
        neck_channels=32,
        embedding_dim=8,
    )


def test_assigner_separates_detection_and_actor_semantics() -> None:
    assigner = AnchorFreeAssigner(input_size=INPUT_SIZE)
    target = _target()

    assignment = assigner(
        [target],
        _strides(),
        prediction_dtype=torch.float32,
        prediction_device=torch.device("cpu"),
        metadata=[{"path": "fall/example"}],
    )

    matched = assignment.matched_gt_indices[0]
    assert set(matched[matched >= 0].tolist()) == {0, 1}
    assert assignment.actor_mask.sum() > 0
    assert torch.all(
        assignment.matched_gt_indices[assignment.actor_mask] == 1
    )
    non_actor = assignment.foreground_mask & ~assignment.actor_mask
    assert torch.all(assignment.stage_targets[non_actor] == -1)
    assert torch.all(assignment.semantic_weights[non_actor] == 0)
    assert torch.all(assignment.stage_targets[assignment.actor_mask] == 1)
    assert torch.all(assignment.action_targets[assignment.actor_mask] == 1)


def test_assigner_forces_a_location_for_tiny_edge_box() -> None:
    assigner = AnchorFreeAssigner(input_size=INPUT_SIZE)
    target = _target(
        boxes=[[0.0, 0.0, 0.005, 0.005]],
        actor_box_index=0,
    )

    assignment = assigner(
        [target],
        _strides(),
        prediction_dtype=torch.float32,
        prediction_device=torch.device("cpu"),
    )

    assert assignment.foreground_mask.sum().item() == 1
    assert assignment.actor_mask.sum().item() == 1


def test_criterion_is_finite_and_backpropagates_all_heads() -> None:
    torch.manual_seed(7)
    model = _small_model().train()
    criterion = EdgeFallCriterion(input_size=INPUT_SIZE)
    images = torch.randn(2, 3, *INPUT_SIZE)
    targets = [_target(), _target()]

    outputs = model(images)
    losses = criterion(
        outputs,
        targets,
        metadata=[{"path": "fall/a"}, {"path": "fall/b"}],
    )
    losses["loss"].backward()

    for name, value in losses.items():
        assert torch.isfinite(value), name
    assert losses["num_positive"].item() > 0
    assert losses["num_actor_positive"].item() > 0
    assert losses["loss_track"].item() > 0
    assert model.head.box.weight.grad is not None
    assert model.head.objectness.weight.grad is not None
    assert model.head.stage.weight.grad is not None
    assert model.head.onset.weight.grad is not None
    assert model.head.embedding.weight.grad is not None
    assert model.head.action.weight.grad is not None


def test_empty_detection_target_only_uses_negative_objectness() -> None:
    model = _small_model().eval()
    criterion = EdgeFallCriterion(input_size=INPUT_SIZE)
    target = _target(
        boxes=[],
        actor_box_index=-1,
        actor_identity=-1,
    )

    outputs = model(torch.randn(1, 3, *INPUT_SIZE))
    losses = criterion(outputs, [target])

    assert losses["num_positive"].item() == 0
    assert losses["num_actor_positive"].item() == 0
    assert losses["loss_objectness"].item() > 0
    assert losses["loss_box"].item() == 0
    assert losses["loss_stage"].item() == 0
    assert losses["loss_onset"].item() == 0
    assert losses["loss_action"].item() == 0
    assert losses["loss_track"].item() == 0
    assert torch.isfinite(losses["loss"])


def test_head_uses_low_positive_priors() -> None:
    model = _small_model()
    expected = math.log(0.01 / 0.99)

    torch.testing.assert_close(
        model.head.objectness.bias,
        torch.full_like(model.head.objectness.bias, expected),
    )
    torch.testing.assert_close(
        model.head.onset.bias,
        torch.full_like(model.head.onset.bias, expected),
    )


def test_bbox_classifier_criterion_has_no_localization_loss() -> None:
    model = BBoxFallClassifier(
        input_size=INPUT_SIZE,
        crop_size=(32, 32),
        backbone_channels=(8, 16, 24, 32, 48),
        visual_dim=32,
        temporal_dim=32,
        history_offsets=(0, 1),
    ).train()
    criterion = BBoxFallCriterion()
    images = torch.randn(2, 2, 3, *INPUT_SIZE)
    boxes = torch.tensor(
        [
            [[0.1, 0.1, 0.5, 0.9], [0.1, 0.1, 0.5, 0.9]],
            [[0.2, 0.3, 0.9, 0.8], [0.2, 0.2, 0.7, 0.9]],
        ]
    )
    valid = torch.ones(2, 2)
    outputs = model.forward_history(images, boxes, valid, valid)
    targets = [_target(actor_box_index=0), _target(actor_box_index=0)]

    losses = criterion(outputs, targets)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert losses["loss_fall"].item() > 0
    assert losses["loss_consistency"].item() >= 0
    assert "loss_box" not in losses
    assert "loss_objectness" not in losses
    assert losses["num_supervised"].item() == 2


def test_bbox_classifier_video_objectives_rank_hard_negative_video() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            onset=0.0,
            action=0.0,
            consistency=0.0,
            video_mil=1.0,
            video_ranking=1.0,
        ),
        video_ranking_margin=0.2,
    )
    outputs = {
        "stage_logits": torch.zeros(4, 3, requires_grad=True),
        "fall_logits": torch.tensor(
            [0.2, 1.0, 1.2, 0.8],
            requires_grad=True,
        ),
        "onset_logits": torch.zeros(4, requires_grad=True),
        "action_logits": torch.zeros(4, 16, requires_grad=True),
    }
    targets = [_target(actor_box_index=0) for _ in range(4)]
    targets[2]["stage_label"] = torch.tensor(0)
    targets[2]["action_label"] = torch.tensor(5)
    targets[2]["onset_target"] = torch.tensor(0.0)
    targets[3]["stage_label"] = torch.tensor(0)
    targets[3]["action_label"] = torch.tensor(6)
    targets[3]["onset_target"] = torch.tensor(0.0)
    metadata = [
        {
            "path": "fall/positive",
            "video_has_fall_event": True,
            "window_action_label": 1,
        },
        {
            "path": "fall/positive",
            "video_has_fall_event": True,
            "window_action_label": 2,
        },
        {
            "path": "lie_down/negative",
            "video_has_fall_event": False,
            "window_action_label": 5,
        },
        {
            "path": "lie_down/negative",
            "video_has_fall_event": False,
            "window_action_label": 6,
        },
    ]

    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()

    assert losses["loss_video_mil"].item() > 0
    torch.testing.assert_close(
        losses["loss_video_ranking"],
        torch.tensor(0.4),
    )
    assert losses["num_video_groups"].item() == 2
    assert losses["num_video_positive_groups"].item() == 1
    assert outputs["fall_logits"].grad is not None


def test_bbox_classifier_event_head_rejects_fallen_only_clip() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=1.0,
            onset=0.0,
            action=0.0,
            consistency=1.0,
            video_mil=0.0,
            video_ranking=0.0,
        ),
    )
    outputs = {
        "stage_logits": torch.tensor(
            [[0.0, 0.0, 2.0]],
            requires_grad=True,
        ),
        "fall_logits": torch.tensor([2.0], requires_grad=True),
        "onset_logits": torch.zeros(1, requires_grad=True),
        "action_logits": torch.zeros(1, 16, requires_grad=True),
    }
    target = _target(actor_box_index=0)
    target["stage_label"] = torch.tensor(2)
    target["action_label"] = torch.tensor(2)
    target["event_target"] = torch.tensor(0.0)

    losses = criterion(outputs, [target])
    losses["loss"].backward()

    assert losses["num_fall_positive"].item() == 0
    assert losses["loss_fall"].item() > 0
    assert losses["loss_consistency"].item() == 0
    assert outputs["fall_logits"].grad is not None
    assert outputs["fall_logits"].grad.item() > 0


def test_v32_controlled_descent_auxiliary_loss_backpropagates() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            onset=0.0,
            action=0.0,
            control=1.0,
            consistency=0.0,
            video_mil=0.0,
            video_ranking=0.0,
        )
    )
    outputs = {
        "stage_logits": torch.zeros(2, 3, requires_grad=True),
        "fall_logits": torch.zeros(2, requires_grad=True),
        "onset_logits": torch.zeros(2, requires_grad=True),
        "action_logits": torch.zeros(2, 16, requires_grad=True),
        "control_logits": torch.zeros(2, requires_grad=True),
    }
    targets = [_target(actor_box_index=0), _target(actor_box_index=0)]
    targets[0]["action_label"] = torch.tensor(5)
    targets[1]["action_label"] = torch.tensor(1)

    losses = criterion(outputs, targets)
    losses["loss"].backward()

    assert losses["loss_control"].item() > 0
    assert outputs["control_logits"].grad is not None
    assert torch.all(outputs["control_logits"].grad.ne(0))


def test_v32_topk_video_pooling_updates_multiple_hard_windows() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            onset=0.0,
            action=0.0,
            control=0.0,
            consistency=0.0,
            video_mil=1.0,
            video_ranking=0.0,
        ),
        video_pooling="topk-mean",
        video_topk=2,
    )
    fall_logits = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
    outputs = {
        "stage_logits": torch.zeros(3, 3, requires_grad=True),
        "fall_logits": fall_logits,
        "onset_logits": torch.zeros(3, requires_grad=True),
        "action_logits": torch.zeros(3, 16, requires_grad=True),
    }
    targets = [_target(actor_box_index=0) for _ in range(3)]
    metadata = [
        {
            "path": "lie_down/negative",
            "video_has_fall_event": False,
            "event_target": False,
        }
        for _ in range(3)
    ]

    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()

    assert criterion._pool_video_logits(fall_logits.detach()).item() == 1.5
    assert fall_logits.grad is not None
    assert fall_logits.grad[0].abs() > 0
    assert fall_logits.grad[1].abs() > 0
    assert fall_logits.grad[2] == 0


def test_v33_geometry_and_event_state_losses_are_independently_supervised() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            event_state=1.0,
            onset=0.0,
            action=0.0,
            control=0.0,
            geometry_stage=1.0,
            geometry_fall=1.0,
            consistency=0.0,
            video_mil=0.0,
            video_ranking=0.0,
        )
    )
    outputs = {
        "stage_logits": torch.zeros(2, 3, requires_grad=True),
        "fall_logits": torch.zeros(2, requires_grad=True),
        "falling_logits": torch.zeros(2, requires_grad=True),
        "fallen_logits": torch.zeros(2, requires_grad=True),
        "onset_logits": torch.zeros(2, requires_grad=True),
        "action_logits": torch.zeros(2, 16, requires_grad=True),
        "geometry_stage_logits": torch.zeros(2, 3, requires_grad=True),
        "geometry_fall_logits": torch.zeros(2, requires_grad=True),
    }
    targets = [_target(actor_box_index=0), _target(actor_box_index=0)]
    targets[1]["stage_label"] = torch.tensor(2)
    targets[1]["action_label"] = torch.tensor(2)

    losses = criterion(outputs, targets)
    losses["loss"].backward()

    assert losses["loss_event_state"].item() > 0
    assert losses["loss_geometry_stage"].item() > 0
    assert losses["loss_geometry_fall"].item() > 0
    assert outputs["falling_logits"].grad is not None
    assert outputs["fallen_logits"].grad is not None
    assert outputs["geometry_stage_logits"].grad is not None
    assert outputs["geometry_fall_logits"].grad is not None


def test_v33_asymmetric_video_pooling_covers_events_and_hard_negatives() -> None:
    criterion = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=0.0,
            fall=0.0,
            onset=0.0,
            action=0.0,
            consistency=0.0,
            video_mil=1.0,
            video_ranking=0.0,
        ),
        video_pooling="asymmetric",
    )
    fall_logits = torch.tensor([2.0, -2.0, 1.0, -3.0], requires_grad=True)
    outputs = {
        "stage_logits": torch.zeros(4, 3, requires_grad=True),
        "fall_logits": fall_logits,
        "onset_logits": torch.zeros(4, requires_grad=True),
        "action_logits": torch.zeros(4, 16, requires_grad=True),
    }
    targets = [_target(actor_box_index=0) for _ in range(4)]
    metadata = [
        {
            "path": "fall/positive",
            "video_has_fall_event": True,
            "event_target": True,
        },
        {
            "path": "fall/positive",
            "video_has_fall_event": True,
            "event_target": True,
        },
        {
            "path": "lying/negative",
            "video_has_fall_event": False,
            "event_target": False,
        },
        {
            "path": "lying/negative",
            "video_has_fall_event": False,
            "event_target": False,
        },
    ]

    losses = criterion(outputs, targets, metadata)
    losses["loss"].backward()

    assert criterion._pool_video_logits(
        fall_logits[:2].detach(), positive=True
    ).item() == 0.0
    assert criterion._pool_video_logits(
        fall_logits[2:].detach(), positive=False
    ).item() == 1.0
    assert fall_logits.grad is not None
    assert fall_logits.grad[0].abs() > 0
    assert fall_logits.grad[1].abs() > 0
    assert fall_logits.grad[2].abs() > 0
    assert fall_logits.grad[3] == 0
