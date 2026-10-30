"""FCOS-style target assignment for the unified EdgeFall-ST head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class AssignmentResult:
    """Dense targets aligned with all flattened prediction locations."""

    matched_gt_indices: Tensor
    foreground_mask: Tensor
    actor_mask: Tensor
    target_boxes: Tensor
    detection_weights: Tensor
    stage_targets: Tensor
    action_targets: Tensor
    onset_targets: Tensor
    semantic_weights: Tensor
    track_identity_targets: Tensor
    track_weights: Tensor


class AnchorFreeAssigner:
    """Assign boxes to center-sampled feature pyramid locations."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (288, 512),
        strides: tuple[int, ...] = (8, 16, 32),
        regression_ranges: tuple[tuple[float, float], ...] = (
            (0.0, 64.0),
            (64.0, 128.0),
            (128.0, math.inf),
        ),
        center_sampling_radius: float = 1.5,
        min_box_confidence: float = 0.0,
    ) -> None:
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError("input_size dimensions must be positive")
        if not strides or any(stride <= 0 for stride in strides):
            raise ValueError("strides must be positive")
        if len(strides) != len(regression_ranges):
            raise ValueError("Each stride needs one regression range")
        if center_sampling_radius <= 0:
            raise ValueError("center_sampling_radius must be positive")
        if not 0.0 <= min_box_confidence <= 1.0:
            raise ValueError("min_box_confidence must be in [0, 1]")
        for lower, upper in regression_ranges:
            if lower < 0 or upper <= lower:
                raise ValueError("Invalid regression range")

        self.input_size = input_size
        self.strides = strides
        self.regression_ranges = regression_ranges
        self.center_sampling_radius = float(center_sampling_radius)
        self.min_box_confidence = float(min_box_confidence)

    def _locations(
        self,
        output_strides: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if output_strides.ndim != 1:
            raise ValueError("outputs['strides'] must be one-dimensional")
        image_h, image_w = self.input_size
        points: list[Tensor] = []
        ranges: list[Tensor] = []
        expected_strides: list[Tensor] = []
        for stride, regression_range in zip(
            self.strides,
            self.regression_ranges,
            strict=True,
        ):
            height = math.ceil(image_h / stride)
            width = math.ceil(image_w / stride)
            y, x = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )
            points.append(
                torch.stack(
                    (
                        (x.reshape(-1) + 0.5) * stride,
                        (y.reshape(-1) + 0.5) * stride,
                    ),
                    dim=-1,
                )
            )
            count = height * width
            ranges.append(
                torch.tensor(
                    regression_range,
                    device=device,
                    dtype=dtype,
                ).expand(count, 2)
            )
            expected_strides.append(
                torch.full(
                    (count,),
                    float(stride),
                    device=device,
                    dtype=dtype,
                )
            )
        point_tensor = torch.cat(points)
        range_tensor = torch.cat(ranges)
        stride_tensor = torch.cat(expected_strides)
        if output_strides.shape != stride_tensor.shape or not torch.equal(
            output_strides.to(device=device, dtype=dtype),
            stride_tensor,
        ):
            raise ValueError(
                "Prediction locations do not match assigner input_size/strides"
            )
        return point_tensor, range_tensor, stride_tensor

    @staticmethod
    def _identity_key(
        target: dict[str, Tensor],
        metadata: dict[str, Any] | None,
        batch_index: int,
    ) -> tuple[str, int] | None:
        identity = int(target["actor_identity"].item())
        if identity < 0:
            identity = int(target["actor_source_track_id"].item())
        if identity < 0:
            return None
        path = (
            str(metadata["path"])
            if metadata is not None and "path" in metadata
            else f"batch:{batch_index}"
        )
        return path, identity

    def __call__(
        self,
        targets: Sequence[dict[str, Tensor]],
        output_strides: Tensor,
        *,
        prediction_dtype: torch.dtype,
        prediction_device: torch.device,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> AssignmentResult:
        if not targets:
            raise ValueError("targets cannot be empty")
        if metadata is not None and len(metadata) != len(targets):
            raise ValueError("metadata and targets must have equal lengths")

        points, ranges, point_strides = self._locations(
            output_strides,
            dtype=prediction_dtype,
            device=prediction_device,
        )
        batch_size = len(targets)
        location_count = points.shape[0]
        matched_gt_indices = torch.full(
            (batch_size, location_count),
            -1,
            dtype=torch.long,
            device=prediction_device,
        )
        target_boxes = torch.zeros(
            (batch_size, location_count, 4),
            dtype=prediction_dtype,
            device=prediction_device,
        )
        detection_weights = torch.zeros(
            (batch_size, location_count),
            dtype=prediction_dtype,
            device=prediction_device,
        )
        actor_mask = torch.zeros(
            (batch_size, location_count),
            dtype=torch.bool,
            device=prediction_device,
        )
        stage_targets = torch.full(
            (batch_size, location_count),
            -1,
            dtype=torch.long,
            device=prediction_device,
        )
        action_targets = torch.full_like(stage_targets, -1)
        onset_targets = torch.zeros(
            (batch_size, location_count),
            dtype=prediction_dtype,
            device=prediction_device,
        )
        semantic_weights = torch.zeros_like(onset_targets)
        track_identity_targets = torch.full_like(stage_targets, -1)
        track_weights = torch.zeros_like(onset_targets)

        identity_ids: dict[tuple[str, int], int] = {}
        image_h, image_w = self.input_size
        image_scale = torch.tensor(
            [image_w, image_h, image_w, image_h],
            dtype=prediction_dtype,
            device=prediction_device,
        )
        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(
                device=prediction_device,
                dtype=prediction_dtype,
            )
            confidences = target["box_confidences"].to(
                device=prediction_device,
                dtype=prediction_dtype,
            )
            if boxes.ndim != 2 or boxes.shape[-1] != 4:
                raise ValueError("target boxes must have shape [N, 4]")
            if confidences.shape != (boxes.shape[0],):
                raise ValueError("box_confidences must have shape [N]")
            valid = (
                torch.isfinite(boxes).all(dim=-1)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
                & (confidences >= self.min_box_confidence)
            )
            valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                continue
            valid_boxes = boxes[valid_indices].clamp(0.0, 1.0)
            valid_confidences = confidences[valid_indices].clamp(0.0, 1.0)
            pixel_boxes = valid_boxes * image_scale

            left = points[:, None, 0] - pixel_boxes[None, :, 0]
            top = points[:, None, 1] - pixel_boxes[None, :, 1]
            right = pixel_boxes[None, :, 2] - points[:, None, 0]
            bottom = pixel_boxes[None, :, 3] - points[:, None, 1]
            distances = torch.stack((left, top, right, bottom), dim=-1)
            inside_box = distances.amin(dim=-1) > 0
            max_distance = distances.amax(dim=-1)
            scale_match = (
                (max_distance >= ranges[:, None, 0])
                & (max_distance <= ranges[:, None, 1])
            )

            centers = torch.stack(
                (
                    (pixel_boxes[:, 0] + pixel_boxes[:, 2]) * 0.5,
                    (pixel_boxes[:, 1] + pixel_boxes[:, 3]) * 0.5,
                ),
                dim=-1,
            )
            radius = self.center_sampling_radius * point_strides[:, None]
            center_match = (
                (points[:, None, 0] >= centers[None, :, 0] - radius)
                & (points[:, None, 0] <= centers[None, :, 0] + radius)
                & (points[:, None, 1] >= centers[None, :, 1] - radius)
                & (points[:, None, 1] <= centers[None, :, 1] + radius)
            )
            eligible = inside_box & scale_match & center_match
            areas = (
                (pixel_boxes[:, 2] - pixel_boxes[:, 0])
                * (pixel_boxes[:, 3] - pixel_boxes[:, 1])
            )
            costs = areas.unsqueeze(0).expand(location_count, -1)
            costs = costs.masked_fill(~eligible, math.inf)
            minimum_cost, local_matches = costs.min(dim=1)
            foreground = torch.isfinite(minimum_cost)
            local_matches = local_matches.masked_fill(~foreground, -1)

            for gt_index in range(valid_boxes.shape[0]):
                if torch.any(local_matches == gt_index):
                    continue
                available = ~foreground
                gt_max_size = torch.maximum(
                    pixel_boxes[gt_index, 2] - pixel_boxes[gt_index, 0],
                    pixel_boxes[gt_index, 3] - pixel_boxes[gt_index, 1],
                )
                preferred_scale = (
                    (gt_max_size >= ranges[:, 0])
                    & (gt_max_size <= ranges[:, 1])
                )
                candidates = available & preferred_scale
                if not torch.any(candidates):
                    candidates = available
                if not torch.any(candidates):
                    raise RuntimeError("No prediction location available for GT")
                normalized_delta = (
                    points - centers[gt_index]
                ) / point_strides[:, None]
                center_distance = normalized_delta.square().sum(dim=-1)
                center_distance = center_distance.masked_fill(
                    ~candidates,
                    math.inf,
                )
                forced_location = int(center_distance.argmin().item())
                local_matches[forced_location] = gt_index
                foreground[forced_location] = True

            selected_local = local_matches[foreground]
            selected_original = valid_indices[selected_local]
            matched_gt_indices[batch_index, foreground] = selected_original
            target_boxes[batch_index, foreground] = valid_boxes[selected_local]
            detection_weights[batch_index, foreground] = valid_confidences[
                selected_local
            ]

            actor_index = int(target["actor_box_index"].item())
            current_actor_mask = foreground.clone()
            current_actor_mask[foreground] = selected_original == actor_index
            actor_mask[batch_index] = current_actor_mask
            if not torch.any(current_actor_mask):
                continue

            stage_targets[batch_index, current_actor_mask] = int(
                target["stage_label"].item()
            )
            action_targets[batch_index, current_actor_mask] = int(
                target["action_label"].item()
            )
            onset_targets[batch_index, current_actor_mask] = float(
                target["onset_target"].item()
            )
            actor_confidences = detection_weights[
                batch_index,
                current_actor_mask,
            ]
            semantic_weights[batch_index, current_actor_mask] = (
                float(target["action_loss_weight"].item())
                * actor_confidences
            )
            track_weights[batch_index, current_actor_mask] = (
                float(target["track_loss_weight"].item())
                * actor_confidences
            )
            identity_key = self._identity_key(
                target,
                None if metadata is None else metadata[batch_index],
                batch_index,
            )
            if identity_key is not None:
                identity_id = identity_ids.setdefault(
                    identity_key,
                    len(identity_ids),
                )
                track_identity_targets[
                    batch_index,
                    current_actor_mask,
                ] = identity_id

        foreground_mask = matched_gt_indices >= 0
        return AssignmentResult(
            matched_gt_indices=matched_gt_indices,
            foreground_mask=foreground_mask,
            actor_mask=actor_mask,
            target_boxes=target_boxes,
            detection_weights=detection_weights,
            stage_targets=stage_targets,
            action_targets=action_targets,
            onset_targets=onset_targets,
            semantic_weights=semantic_weights,
            track_identity_targets=track_identity_targets,
            track_weights=track_weights,
        )
