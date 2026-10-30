"""Shared runtime for external V3.4 frozen-YOLO evaluations."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from edgefall_st.data.dataset import (
    _letterbox_image_uint8,
    _letterbox_parameters,
    _transform_normalized_box,
)
from edgefall_st.features.adapter import UltralyticsYOLOFeatureAdapter
from edgefall_st.features.cache import roi_feature_shape
from edgefall_st.models.fygr import (
    FYGR_FUSION_CONCAT_MLP,
    FrozenYoloGlobalROIConfirmer,
)


NormalizedBox = tuple[float, float, float, float]
FrameItem = tuple[int, np.ndarray, NormalizedBox | None]


@dataclass(frozen=True)
class ExternalFeatureSequence:
    global_features: Tensor
    roi_features: Tensor
    geometry_boxes: Tensor
    visual_boxes: Tensor
    actor_valid_mask: Tensor
    decoded_mask: Tensor


@dataclass(frozen=True)
class ExternalWindowScores:
    sample_indices: tuple[int, ...]
    score: Tensor
    geometry_score: Tensor
    global_score: Tensor | None
    roi_score: Tensor | None
    stage_probabilities: Tensor
    onset_score: Tensor


class FygrExternalRuntime:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        weights_path: Path,
        device: torch.device,
        amp: bool,
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("version") not in {
            "v3.4-fygr",
            "v3.5-md-fygr",
            "v3.6-gcva",
            "v3.6-md-gcva",
            "v3.7-person-mil",
            "v3.7-md-person-mil",
            "v3.8-temporal-token",
            "v3.8-md-temporal-token",
            "v3.8-person-mil-temporal-token",
            "v3.8-md-person-mil-temporal-token",
            "v3.9-summary-ranking",
            "v3.9-md-summary-ranking",
            "v3.10-hard-tail-ranking",
            "v3.10-md-hard-tail-ranking",
        }:
            raise ValueError("Checkpoint is not a supported FYGR checkpoint")
        arguments = argparse.Namespace(**checkpoint["arguments"])
        contract = checkpoint["feature_contract"]
        if _sha256(weights_path) != contract["weights_sha256"]:
            raise ValueError("YOLO weights do not match the checkpoint contract")

        adapter_spec = contract["feature_adapter"]
        self.adapter = UltralyticsYOLOFeatureAdapter(
            weights_path,
            global_levels=tuple(adapter_spec["global_levels"]),
            roi_levels=tuple(adapter_spec["roi_levels"]),
            roi_output_size=int(adapter_spec["roi_output_size"]),
            roi_context_scale=float(adapter_spec["roi_context_scale"]),
        ).to(device)
        self.adapter.eval()
        arrays = contract["arrays"]
        self.model = FrozenYoloGlobalROIConfirmer(
            geometry_checkpoint=Path(arguments.geometry_checkpoint),
            history_offsets=tuple(contract["history_offsets"]),
            visual_offsets=tuple(contract["visual_offsets"]),
            global_feature_dim=int(arrays["global_features"][-1]),
            roi_feature_shape=roi_feature_shape(contract),
            temporal_dim=int(arguments.temporal_dim),
            dropout=float(arguments.dropout),
            use_global=bool(arguments.use_global),
            use_roi=bool(arguments.use_roi),
            fusion_mode=str(
                getattr(arguments, "fusion_mode", FYGR_FUSION_CONCAT_MLP)
            ),
            attention_heads=int(getattr(arguments, "attention_heads", 4)),
            attention_ffn_dim=int(
                getattr(arguments, "attention_ffn_dim", 256)
            ),
            person_mil_temperature=float(
                getattr(arguments, "person_mil_temperature", 0.0)
            ),
        ).to(device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        self.weights_path = weights_path
        # V3.4 checkpoints store the human-facing one-based epoch number.
        self.checkpoint_epoch = int(checkpoint["epoch"])
        self.contract = contract
        self.device = device
        self.history_offsets = tuple(int(value) for value in contract["history_offsets"])
        self.visual_offsets = tuple(int(value) for value in contract["visual_offsets"])
        self.visual_input_size = tuple(int(value) for value in contract["visual_input_size"])
        self.geometry_input_size = tuple(int(value) for value in contract["geometry_input_size"])
        self.effective_fps = float(contract["effective_fps"])
        self.global_shape = tuple(int(value) for value in arrays["global_features"])
        self.roi_shape = roi_feature_shape(contract)
        self.feature_amp_enabled = bool(amp and device.type == "cuda")
        self.model_amp_enabled = bool(amp and device.type == "cuda")
        self.model_amp_dtype = (
            torch.bfloat16
            if self.model_amp_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    @property
    def parameter_count_without_yolo(self) -> int:
        return self.model.parameter_count

    @property
    def trainable_parameter_count(self) -> int:
        return self.model.trainable_parameter_count

    def close(self) -> None:
        self.adapter.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _transformed_box(
    box: NormalizedBox | None,
    *,
    source_size: tuple[int, int],
    input_size: tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
) -> NormalizedBox | None:
    transformed = _transform_normalized_box(
        None if box is None else list(box),
        source_size=source_size,
        input_size=input_size,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    if transformed is None:
        return None
    return tuple(float(value) for value in transformed)


def extract_external_sequence(
    runtime: FygrExternalRuntime,
    frames: Iterable[FrameItem],
    *,
    sample_count: int,
    feature_batch_size: int,
) -> ExternalFeatureSequence:
    if sample_count <= 0 or feature_batch_size <= 0:
        raise ValueError("sample_count and feature_batch_size must be positive")
    global_features = torch.zeros((sample_count, *runtime.global_shape))
    roi_features = torch.zeros((sample_count, *runtime.roi_shape))
    geometry_boxes = torch.zeros((sample_count, 4))
    visual_boxes = torch.zeros((sample_count, 4))
    actor_valid = torch.zeros(sample_count)
    decoded = torch.zeros(sample_count)
    seen: set[int] = set()
    pending_images: list[Tensor] = []
    pending_boxes: list[NormalizedBox] = []
    pending_valid: list[float] = []
    pending_positions: list[int] = []

    def flush() -> None:
        if not pending_images:
            return
        images = torch.stack(pending_images).to(
            runtime.device, non_blocking=True
        ).float().div_(255.0)
        boxes = torch.tensor(
            pending_boxes, device=runtime.device, dtype=torch.float32
        )
        mask = torch.tensor(
            pending_valid, device=runtime.device, dtype=torch.float32
        )
        with torch.inference_mode(), torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.float16,
            enabled=runtime.feature_amp_enabled,
        ):
            output = runtime.adapter(images, boxes, mask)
        rows = torch.tensor(pending_positions, dtype=torch.long)
        current_global = output.global_features.float().cpu()
        current_roi = output.roi_features.float().cpu()
        if tuple(current_global.shape[1:]) != runtime.global_shape:
            raise ValueError("Global feature shape violates checkpoint contract")
        if tuple(current_roi.shape[1:]) != runtime.roi_shape:
            raise ValueError("ROI feature shape violates checkpoint contract")
        global_features[rows] = current_global
        roi_features[rows] = current_roi
        pending_images.clear()
        pending_boxes.clear()
        pending_valid.clear()
        pending_positions.clear()

    for sample_index, image_bgr, source_box in frames:
        if sample_index < 0 or sample_index >= sample_count:
            raise IndexError(f"Sample index out of range: {sample_index}")
        if sample_index in seen:
            raise ValueError(f"Duplicate sampled frame: {sample_index}")
        seen.add(sample_index)
        source_size = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))
        visual_image, visual_scale, visual_left, visual_top = (
            _letterbox_image_uint8(
                image_bgr, runtime.visual_input_size, fill_value=114
            )
        )
        visual_box = _transformed_box(
            source_box,
            source_size=source_size,
            input_size=runtime.visual_input_size,
            scale=visual_scale,
            pad_left=visual_left,
            pad_top=visual_top,
        )
        geometry_scale, geometry_left, geometry_top = _letterbox_parameters(
            source_size, runtime.geometry_input_size
        )
        geometry_box = _transformed_box(
            source_box,
            source_size=source_size,
            input_size=runtime.geometry_input_size,
            scale=geometry_scale,
            pad_left=geometry_left,
            pad_top=geometry_top,
        )
        valid = float(visual_box is not None and geometry_box is not None)
        decoded[sample_index] = 1.0
        actor_valid[sample_index] = valid
        if geometry_box is not None:
            geometry_boxes[sample_index] = torch.tensor(geometry_box)
        if visual_box is not None:
            visual_boxes[sample_index] = torch.tensor(visual_box)
        pending_images.append(visual_image)
        pending_boxes.append(
            visual_box if visual_box is not None else (0.0, 0.0, 0.0, 0.0)
        )
        pending_valid.append(valid)
        pending_positions.append(sample_index)
        if len(pending_images) >= feature_batch_size:
            flush()
    flush()
    return ExternalFeatureSequence(
        global_features=global_features,
        roi_features=roi_features,
        geometry_boxes=geometry_boxes,
        visual_boxes=visual_boxes,
        actor_valid_mask=actor_valid,
        decoded_mask=decoded,
    )


def evaluate_external_windows(
    runtime: FygrExternalRuntime,
    sequence: ExternalFeatureSequence,
    current_indices: Sequence[int],
    *,
    window_batch_size: int,
) -> ExternalWindowScores:
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")
    current_values = tuple(int(value) for value in current_indices)
    score_parts: list[Tensor] = []
    geometry_parts: list[Tensor] = []
    global_parts: list[Tensor] = []
    roi_parts: list[Tensor] = []
    stage_parts: list[Tensor] = []
    onset_parts: list[Tensor] = []
    history_offsets = torch.tensor(runtime.history_offsets, dtype=torch.long)
    visual_offsets = torch.tensor(runtime.visual_offsets, dtype=torch.long)

    with torch.inference_mode():
        for start in range(0, len(current_values), window_batch_size):
            current = torch.tensor(
                current_values[start : start + window_batch_size],
                dtype=torch.long,
            )
            history_indices = current[:, None] - history_offsets[None]
            history_valid = history_indices.ge(0)
            safe_history = history_indices.clamp_min(0)
            history_valid &= sequence.decoded_mask[safe_history].gt(0)
            visual_indices = current[:, None] - visual_offsets[None]
            safe_visual = visual_indices.clamp_min(0)

            global_values = sequence.global_features[safe_visual].to(
                runtime.device, non_blocking=True
            )
            roi_values = sequence.roi_features[safe_visual].to(
                runtime.device, non_blocking=True
            )
            boxes = sequence.geometry_boxes[safe_history].to(
                runtime.device, non_blocking=True
            )
            valid_mask = history_valid.to(
                runtime.device, dtype=torch.float32, non_blocking=True
            )
            actor_mask = (
                sequence.actor_valid_mask[safe_history]
                * history_valid.float()
            ).to(runtime.device, non_blocking=True)
            with torch.autocast(
                device_type=runtime.device.type,
                dtype=runtime.model_amp_dtype,
                enabled=runtime.model_amp_enabled,
            ):
                outputs = runtime.model(
                    global_values,
                    roi_values,
                    boxes,
                    valid_mask,
                    actor_mask,
                )
            score_parts.append(outputs["fall_logits"].float().sigmoid().cpu())
            geometry_parts.append(
                outputs["geometry_fall_logits"].float().sigmoid().cpu()
            )
            if "global_fall_logits" in outputs:
                global_parts.append(
                    outputs["global_fall_logits"].float().sigmoid().cpu()
                )
            if "roi_fall_logits" in outputs:
                roi_parts.append(
                    outputs["roi_fall_logits"].float().sigmoid().cpu()
                )
            stage_parts.append(outputs["stage_logits"].float().softmax(-1).cpu())
            onset_parts.append(outputs["onset_logits"].float().sigmoid().cpu())

    def concatenate(parts: list[Tensor]) -> Tensor:
        return torch.cat(parts) if parts else torch.empty(0)

    return ExternalWindowScores(
        sample_indices=current_values,
        score=concatenate(score_parts),
        geometry_score=concatenate(geometry_parts),
        global_score=concatenate(global_parts) if global_parts else None,
        roi_score=concatenate(roi_parts) if roi_parts else None,
        stage_probabilities=(
            torch.cat(stage_parts) if stage_parts else torch.empty((0, 3))
        ),
        onset_score=concatenate(onset_parts),
    )


def score_columns(scores: ExternalWindowScores) -> dict[str, Tensor]:
    result = {
        "final": scores.score,
        "geometry": scores.geometry_score,
    }
    if scores.global_score is not None:
        result["global"] = scores.global_score
    if scores.roi_score is not None:
        result["roi"] = scores.roi_score
    return result
