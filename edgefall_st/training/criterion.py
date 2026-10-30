"""Multi-task criterion for jointly training EdgeFall-ST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from edgefall_st.training.assignment import (
    AnchorFreeAssigner,
    AssignmentResult,
)


@dataclass(frozen=True)
class LossWeights:
    box: float = 5.0
    objectness: float = 1.0
    stage: float = 2.0
    onset: float = 1.0
    action: float = 0.5
    track: float = 0.1


def _sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float,
    gamma: float,
) -> Tensor:
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    target_probability = (
        probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    )
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (
        alpha_factor
        * (1.0 - target_probability).pow(gamma)
        * cross_entropy
    )


def _complete_iou_loss(predicted: Tensor, target: Tensor) -> Tensor:
    predicted_width = (predicted[:, 2] - predicted[:, 0]).clamp_min(1e-6)
    predicted_height = (predicted[:, 3] - predicted[:, 1]).clamp_min(1e-6)
    target_width = (target[:, 2] - target[:, 0]).clamp_min(1e-6)
    target_height = (target[:, 3] - target[:, 1]).clamp_min(1e-6)

    intersection_left = torch.maximum(predicted[:, 0], target[:, 0])
    intersection_top = torch.maximum(predicted[:, 1], target[:, 1])
    intersection_right = torch.minimum(predicted[:, 2], target[:, 2])
    intersection_bottom = torch.minimum(predicted[:, 3], target[:, 3])
    intersection = (
        (intersection_right - intersection_left).clamp_min(0.0)
        * (intersection_bottom - intersection_top).clamp_min(0.0)
    )
    union = (
        predicted_width * predicted_height
        + target_width * target_height
        - intersection
    ).clamp_min(1e-6)
    iou = intersection / union

    predicted_center_x = (predicted[:, 0] + predicted[:, 2]) * 0.5
    predicted_center_y = (predicted[:, 1] + predicted[:, 3]) * 0.5
    target_center_x = (target[:, 0] + target[:, 2]) * 0.5
    target_center_y = (target[:, 1] + target[:, 3]) * 0.5
    center_distance = (
        (predicted_center_x - target_center_x).square()
        + (predicted_center_y - target_center_y).square()
    )
    enclosing_width = (
        torch.maximum(predicted[:, 2], target[:, 2])
        - torch.minimum(predicted[:, 0], target[:, 0])
    )
    enclosing_height = (
        torch.maximum(predicted[:, 3], target[:, 3])
        - torch.minimum(predicted[:, 1], target[:, 1])
    )
    enclosing_diagonal = (
        enclosing_width.square() + enclosing_height.square()
    ).clamp_min(1e-6)
    aspect_penalty = (
        4.0
        / torch.pi**2
        * (
            torch.atan(target_width / target_height)
            - torch.atan(predicted_width / predicted_height)
        ).square()
    )
    with torch.no_grad():
        aspect_weight = aspect_penalty / (
            1.0 - iou + aspect_penalty
        ).clamp_min(1e-6)
    complete_iou = (
        iou
        - center_distance / enclosing_diagonal
        - aspect_weight * aspect_penalty
    )
    return 1.0 - complete_iou


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.numel() == 0:
        return weights.sum() * 0.0
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _supervised_contrastive_loss(
    embeddings: Tensor,
    identities: Tensor,
    weights: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    similarities = embeddings @ embeddings.transpose(0, 1) / temperature
    similarities = similarities - similarities.amax(dim=1, keepdim=True).detach()
    self_mask = ~torch.eye(
        embeddings.shape[0],
        dtype=torch.bool,
        device=embeddings.device,
    )
    positive_mask = (
        identities[:, None] == identities[None, :]
    ) & self_mask
    positive_counts = positive_mask.sum(dim=1)
    valid_anchors = positive_counts > 0
    if not torch.any(valid_anchors):
        return embeddings.sum() * 0.0

    exp_similarities = similarities.exp() * self_mask
    log_probabilities = similarities - torch.log(
        exp_similarities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive_log_probability = (
        (log_probabilities * positive_mask).sum(dim=1)
        / positive_counts.clamp_min(1)
    )
    return _weighted_mean(
        -mean_positive_log_probability[valid_anchors],
        weights[valid_anchors],
    )


class EdgeFallCriterion(nn.Module):
    """Compute detection and actor-specific semantic losses."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (288, 512),
        assigner: AnchorFreeAssigner | None = None,
        loss_weights: LossWeights = LossWeights(),
        objectness_alpha: float = 0.25,
        objectness_gamma: float = 2.0,
        stage_gamma: float = 2.0,
        onset_alpha: float = 0.75,
        onset_gamma: float = 2.0,
        stage_class_weights: Sequence[float] = (1.0, 2.0, 2.0),
        action_class_weights: Sequence[float] | None = None,
        track_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 <= objectness_alpha <= 1.0:
            raise ValueError("objectness_alpha must be in [0, 1]")
        if not 0.0 <= onset_alpha <= 1.0:
            raise ValueError("onset_alpha must be in [0, 1]")
        if objectness_gamma < 0 or stage_gamma < 0 or onset_gamma < 0:
            raise ValueError("Focal gamma values cannot be negative")
        if track_temperature <= 0:
            raise ValueError("track_temperature must be positive")
        if len(stage_class_weights) != 3:
            raise ValueError("stage_class_weights must have three values")
        action_weights = (
            tuple(action_class_weights)
            if action_class_weights is not None
            else (1.0,) * 16
        )
        if len(action_weights) != 16:
            raise ValueError("action_class_weights must have 16 values")

        self.assigner = assigner or AnchorFreeAssigner(input_size=input_size)
        self.loss_weights = loss_weights
        self.objectness_alpha = float(objectness_alpha)
        self.objectness_gamma = float(objectness_gamma)
        self.stage_gamma = float(stage_gamma)
        self.onset_alpha = float(onset_alpha)
        self.onset_gamma = float(onset_gamma)
        self.track_temperature = float(track_temperature)
        self.register_buffer(
            "stage_class_weights",
            torch.tensor(stage_class_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "action_class_weights",
            torch.tensor(action_weights, dtype=torch.float32),
        )

    @staticmethod
    def _validate_outputs(outputs: dict[str, Tensor], batch_size: int) -> None:
        required = {
            "boxes",
            "objectness_logits",
            "stage_logits",
            "onset_logits",
            "track_embeddings",
            "action_logits",
            "strides",
        }
        missing = required.difference(outputs)
        if missing:
            raise ValueError(f"Model outputs are missing: {sorted(missing)}")
        boxes = outputs["boxes"]
        if boxes.ndim != 3 or boxes.shape[0] != batch_size:
            raise ValueError("outputs['boxes'] must have shape [B, A, 4]")
        locations = boxes.shape[1]
        expected_shapes = {
            "objectness_logits": (batch_size, locations),
            "onset_logits": (batch_size, locations),
        }
        for name, shape in expected_shapes.items():
            if outputs[name].shape != shape:
                raise ValueError(f"outputs['{name}'] must have shape {shape}")
        if outputs["stage_logits"].shape != (batch_size, locations, 3):
            raise ValueError("stage_logits must have shape [B, A, 3]")
        if outputs["action_logits"].shape != (batch_size, locations, 16):
            raise ValueError("action_logits must have shape [B, A, 16]")
        if outputs["track_embeddings"].ndim != 3 or outputs[
            "track_embeddings"
        ].shape[:2] != (batch_size, locations):
            raise ValueError("track_embeddings must have shape [B, A, D]")

    def _stage_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        weights: Tensor,
    ) -> Tensor:
        if logits.shape[0] == 0:
            return logits.sum() * 0.0
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.stage_class_weights.to(logits.dtype),
            reduction="none",
        )
        probabilities = logits.softmax(dim=-1)
        target_probabilities = probabilities.gather(
            1,
            targets[:, None],
        ).squeeze(1)
        return _weighted_mean(
            (1.0 - target_probabilities).pow(self.stage_gamma)
            * cross_entropy,
            weights,
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Tensor]:
        self._validate_outputs(outputs, len(targets))
        # Loss arithmetic stays in FP32 under autocast. In particular, CIoU
        # contains small squared distances that are unsafe in FP16.
        prediction = outputs["boxes"].float()
        objectness_logits = outputs["objectness_logits"].float()
        stage_logits = outputs["stage_logits"].float()
        onset_logits_all = outputs["onset_logits"].float()
        action_logits_all = outputs["action_logits"].float()
        track_embeddings = outputs["track_embeddings"].float()
        assignment: AssignmentResult = self.assigner(
            targets,
            outputs["strides"],
            prediction_dtype=prediction.dtype,
            prediction_device=prediction.device,
            metadata=metadata,
        )
        foreground = assignment.foreground_mask
        actor = assignment.actor_mask & (assignment.semantic_weights > 0)

        objectness_targets = foreground.to(prediction.dtype)
        objectness_values = _sigmoid_focal_loss(
            objectness_logits,
            objectness_targets,
            alpha=self.objectness_alpha,
            gamma=self.objectness_gamma,
        )
        objectness_weights = torch.ones_like(objectness_values)
        objectness_weights[foreground] = assignment.detection_weights[
            foreground
        ]
        positive_normalizer = foreground.sum().clamp_min(1).to(prediction.dtype)
        loss_objectness = (
            objectness_values * objectness_weights
        ).sum() / positive_normalizer

        predicted_positive_boxes = prediction[foreground]
        target_positive_boxes = assignment.target_boxes[foreground]
        box_weights = assignment.detection_weights[foreground]
        ciou_values = _complete_iou_loss(
            predicted_positive_boxes,
            target_positive_boxes,
        )
        l1_values = F.l1_loss(
            predicted_positive_boxes,
            target_positive_boxes,
            reduction="none",
        ).sum(dim=-1)
        loss_box_ciou = _weighted_mean(ciou_values, box_weights)
        loss_box_l1 = _weighted_mean(l1_values, box_weights)
        loss_box = loss_box_ciou + loss_box_l1

        semantic_weights = assignment.semantic_weights[actor]
        loss_stage = self._stage_loss(
            stage_logits[actor],
            assignment.stage_targets[actor],
            semantic_weights,
        )
        action_logits = action_logits_all[actor]
        if action_logits.shape[0] == 0:
            loss_action = action_logits.sum() * 0.0
        else:
            action_values = F.cross_entropy(
                action_logits,
                assignment.action_targets[actor],
                weight=self.action_class_weights.to(action_logits.dtype),
                reduction="none",
            )
            loss_action = _weighted_mean(action_values, semantic_weights)

        onset_logits = onset_logits_all[actor]
        if onset_logits.shape[0] == 0:
            loss_onset = onset_logits.sum() * 0.0
        else:
            onset_values = _sigmoid_focal_loss(
                onset_logits,
                assignment.onset_targets[actor],
                alpha=self.onset_alpha,
                gamma=self.onset_gamma,
            )
            loss_onset = _weighted_mean(onset_values, semantic_weights)

        track_mask = (
            assignment.actor_mask
            & (assignment.track_identity_targets >= 0)
            & (assignment.track_weights > 0)
        )
        loss_track = _supervised_contrastive_loss(
            track_embeddings[track_mask],
            assignment.track_identity_targets[track_mask],
            assignment.track_weights[track_mask],
            temperature=self.track_temperature,
        )

        loss = (
            self.loss_weights.box * loss_box
            + self.loss_weights.objectness * loss_objectness
            + self.loss_weights.stage * loss_stage
            + self.loss_weights.onset * loss_onset
            + self.loss_weights.action * loss_action
            + self.loss_weights.track * loss_track
        )
        return {
            "loss": loss,
            "loss_box": loss_box,
            "loss_box_ciou": loss_box_ciou,
            "loss_box_l1": loss_box_l1,
            "loss_objectness": loss_objectness,
            "loss_stage": loss_stage,
            "loss_onset": loss_onset,
            "loss_action": loss_action,
            "loss_track": loss_track,
            "num_positive": foreground.sum().detach(),
            "num_actor_positive": actor.sum().detach(),
        }
