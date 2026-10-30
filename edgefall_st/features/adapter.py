"""Unified adapters for extracting frozen global and actor ROI features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import roi_align


@dataclass(frozen=True)
class FrozenFeatureBatch:
    """Compact visual descriptors produced by one frozen detector forward."""

    global_features: Tensor
    roi_features: Tensor
    actor_valid_mask: Tensor


def _expanded_normalized_boxes(boxes: Tensor, context_scale: float) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    left, top, right, bottom = boxes.clamp(0.0, 1.0).unbind(dim=-1)
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    half_width = (right - left).clamp_min(1e-4) * context_scale * 0.5
    half_height = (bottom - top).clamp_min(1e-4) * context_scale * 0.5
    expanded = torch.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    return expanded


class FrozenPyramidFeatureAdapter(nn.Module):
    """Extract a stable feature contract from an arbitrary detector pyramid.

    Inputs are RGB images already letterboxed to the detector input size and
    normalized to ``[0, 1]``. Actor boxes use normalized XYXY coordinates in
    that same letterboxed coordinate system.
    """

    def __init__(
        self,
        detector: nn.Module,
        *,
        feature_modules: Mapping[str, nn.Module],
        global_levels: Sequence[str] = ("p4", "p5"),
        roi_levels: Sequence[str] = ("p3", "p4"),
        roi_output_size: int = 3,
        roi_context_scale: float = 1.25,
        sampling_ratio: int = 2,
    ) -> None:
        super().__init__()
        if roi_output_size <= 0:
            raise ValueError("roi_output_size must be positive")
        if roi_context_scale < 1.0:
            raise ValueError("roi_context_scale must be at least 1")
        if not feature_modules:
            raise ValueError("feature_modules cannot be empty")
        known_levels = set(feature_modules)
        for name, levels in (
            ("global_levels", global_levels),
            ("roi_levels", roi_levels),
        ):
            missing = set(levels).difference(known_levels)
            if missing:
                raise ValueError(f"{name} contains unknown levels: {missing}")

        self.detector = detector.eval()
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        self.feature_modules = dict(feature_modules)
        self.global_levels = tuple(str(value) for value in global_levels)
        self.roi_levels = tuple(str(value) for value in roi_levels)
        self.roi_output_size = int(roi_output_size)
        self.roi_context_scale = float(roi_context_scale)
        self.sampling_ratio = int(sampling_ratio)
        self._captured: dict[str, Tensor] = {}
        self._hooks = [
            module.register_forward_hook(self._capture_hook(level))
            for level, module in self.feature_modules.items()
        ]

    def _capture_hook(self, level: str):
        def capture(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not isinstance(output, Tensor) or output.ndim != 4:
                raise TypeError(
                    f"Feature level {level!r} must produce a BCHW tensor"
                )
            self._captured[level] = output

        return capture

    def train(self, mode: bool = True) -> FrozenPyramidFeatureAdapter:
        super().train(False)
        self.detector.eval()
        return self

    def close(self) -> None:
        while self._hooks:
            self._hooks.pop().remove()

    @property
    def level_names(self) -> tuple[str, ...]:
        return tuple(self.feature_modules)

    def feature_spec(self) -> dict[str, Any]:
        channels = {
            level: int(feature.shape[1])
            for level, feature in self._captured.items()
        }
        return {
            "adapter": type(self).__name__,
            "levels": list(self.level_names),
            "channels": channels,
            "global_levels": list(self.global_levels),
            "roi_levels": list(self.roi_levels),
            "roi_output_size": self.roi_output_size,
            "roi_context_scale": self.roi_context_scale,
            "coordinate_system": "letterboxed_normalized_xyxy",
        }

    def _roi_features(
        self,
        feature: Tensor,
        boxes: Tensor,
        actor_valid_mask: Tensor,
    ) -> Tensor:
        count, _, feature_h, feature_w = feature.shape
        multi_person = boxes.ndim == 3
        if multi_person:
            if boxes.shape[0] != count or boxes.shape[-1] != 4:
                raise ValueError("boxes must have shape [N, P, 4]")
            if actor_valid_mask.shape != boxes.shape[:2]:
                raise ValueError(
                    "actor_valid_mask must have shape [N, P]"
                )
            person_count = int(boxes.shape[1])
            flat_boxes = boxes.reshape(count * person_count, 4)
            flat_valid = actor_valid_mask.reshape(count * person_count)
        else:
            if boxes.shape != (count, 4):
                raise ValueError("boxes must have shape [N, 4]")
            if actor_valid_mask.shape != (count,):
                raise ValueError("actor_valid_mask must have shape [N]")
            person_count = 1
            flat_boxes = boxes
            flat_valid = actor_valid_mask
        expanded = _expanded_normalized_boxes(
            flat_boxes.to(dtype=feature.dtype),
            self.roi_context_scale,
        )
        scale = expanded.new_tensor(
            (feature_w, feature_h, feature_w, feature_h)
        )
        feature_boxes = expanded * scale
        valid = flat_valid.gt(0)
        fallback = feature_boxes.new_tensor((0.0, 0.0, 1.0, 1.0))
        feature_boxes = torch.where(valid[:, None], feature_boxes, fallback)
        batch_indices = torch.arange(
            count,
            device=feature.device,
            dtype=feature_boxes.dtype,
        ).repeat_interleave(person_count)[:, None]
        regions = torch.cat((batch_indices, feature_boxes), dim=1)
        pooled = roi_align(
            feature,
            regions,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=self.sampling_ratio,
            aligned=True,
        )
        pooled = pooled * valid[:, None, None, None].to(pooled.dtype)
        if multi_person:
            return pooled.reshape(
                count,
                person_count,
                pooled.shape[1],
                pooled.shape[2],
                pooled.shape[3],
            )
        return pooled

    def clear_captured_features(self) -> None:
        """Discard feature maps captured by an earlier detector forward."""

        self._captured.clear()

    def features_from_captured(
        self,
        actor_boxes: Tensor,
        actor_valid_mask: Tensor,
    ) -> FrozenFeatureBatch:
        """Pool descriptors from the most recent detector forward.

        This split is used by online inference: Ultralytics performs detection
        and preprocessing once, then the selected actor box is pooled from the
        feature maps captured during that same forward.
        """

        missing = set(self.feature_modules).difference(self._captured)
        if missing:
            raise RuntimeError(f"Detector did not produce levels: {missing}")
        first = self._captured[next(iter(self.feature_modules))]
        count = int(first.shape[0])
        if actor_boxes.ndim == 2:
            expected_mask = (count,)
            expected_boxes = (count, 4)
        elif actor_boxes.ndim == 3:
            expected_mask = actor_boxes.shape[:2]
            expected_boxes = (count, actor_boxes.shape[1], 4)
        else:
            raise ValueError(
                "actor_boxes must have shape [N, 4] or [N, P, 4]"
            )
        if actor_boxes.shape != expected_boxes:
            raise ValueError(
                "actor_boxes must have shape [N, 4] or [N, P, 4]"
            )
        if actor_valid_mask.shape != expected_mask:
            raise ValueError("actor_valid_mask shape disagrees with actor_boxes")
        global_features = torch.cat(
            [
                F.adaptive_avg_pool2d(self._captured[level], 1).flatten(1)
                for level in self.global_levels
            ],
            dim=1,
        )
        roi_features = torch.cat(
            [
                self._roi_features(
                    self._captured[level],
                    actor_boxes,
                    actor_valid_mask,
                )
                for level in self.roi_levels
            ],
            dim=2 if actor_boxes.ndim == 3 else 1,
        )
        return FrozenFeatureBatch(
            global_features=global_features,
            roi_features=roi_features,
            actor_valid_mask=actor_valid_mask,
        )

    def forward(
        self,
        images: Tensor,
        actor_boxes: Tensor,
        actor_valid_mask: Tensor,
    ) -> FrozenFeatureBatch:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [N, 3, H, W]")
        if actor_boxes.ndim not in (2, 3) or actor_boxes.shape[0] != images.shape[0]:
            raise ValueError(
                "actor_boxes must have shape [N, 4] or [N, P, 4]"
            )
        if actor_boxes.shape[-1] != 4:
            raise ValueError(
                "actor_boxes must have shape [N, 4] or [N, P, 4]"
            )
        if actor_valid_mask.shape != actor_boxes.shape[:-1]:
            raise ValueError("actor_valid_mask shape disagrees with actor_boxes")

        self.clear_captured_features()
        self.detector(images)
        return self.features_from_captured(actor_boxes, actor_valid_mask)


class UltralyticsYOLOFeatureAdapter(FrozenPyramidFeatureAdapter):
    """Auto-discover P3/P4/P5 inputs to an Ultralytics Detect head."""

    def __init__(
        self,
        weights: str | Path,
        *,
        global_levels: Sequence[str] = ("p4", "p5"),
        roi_levels: Sequence[str] = ("p3", "p4"),
        roi_output_size: int = 3,
        roi_context_scale: float = 1.25,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required; install edgefall-st[pseudo]"
            ) from error

        weights_path = Path(weights)
        wrapper = YOLO(str(weights_path))
        detector = wrapper.model
        modules = getattr(detector, "model", None)
        if not isinstance(modules, nn.Sequential):
            raise TypeError("Unsupported Ultralytics detector structure")
        detect = modules[-1]
        source_indices = getattr(detect, "f", None)
        if not isinstance(source_indices, list) or len(source_indices) != 3:
            raise ValueError(
                "Expected a three-level Ultralytics Detect head"
            )
        level_modules = {
            level: modules[int(index)]
            for level, index in zip(
                ("p3", "p4", "p5"), source_indices, strict=True
            )
        }
        super().__init__(
            detector,
            feature_modules=level_modules,
            global_levels=global_levels,
            roi_levels=roi_levels,
            roi_output_size=roi_output_size,
            roi_context_scale=roi_context_scale,
        )
        # Ultralytics YOLO subclasses nn.Module but overloads train() as a
        # training entry point. Keep it out of PyTorch's child-module tree so
        # adapter.eval() cannot accidentally launch a trainer.
        object.__setattr__(self, "yolo", wrapper)
        self.weights_path = weights_path
        self.source_layer_indices = tuple(int(value) for value in source_indices)

    def feature_spec(self) -> dict[str, Any]:
        result = super().feature_spec()
        result.update(
            {
                "source_layer_indices": list(self.source_layer_indices),
            }
        )
        return result
