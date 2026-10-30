"""Unified person, fall-stage, onset, and association head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from edgefall_st.models.layers import ConvBNAct


class UnifiedAnchorFreeHead(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        input_size: tuple[int, int] = (288, 512),
        strides: tuple[int, int, int] = (8, 16, 32),
        embedding_dim: int = 8,
        num_stages: int = 3,
        num_actions: int = 16,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.strides = strides
        self.embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.num_actions = num_actions
        self.stem = nn.Sequential(
            ConvBNAct(channels, channels, 3, activation=activation),
            ConvBNAct(channels, channels, 3, activation=activation),
        )
        self.box = nn.Conv2d(channels, 4, 1)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.stage = nn.Conv2d(channels, num_stages, 1)
        self.onset = nn.Conv2d(channels, 1, 1)
        self.embedding = nn.Conv2d(channels, embedding_dim, 1)
        self.action = nn.Conv2d(channels, num_actions, 1)
        prior_logit = math.log(0.01 / 0.99)
        nn.init.constant_(self.objectness.bias, prior_logit)
        nn.init.constant_(self.onset.bias, prior_logit)

    @staticmethod
    def _flatten(values: Tensor) -> Tensor:
        return values.flatten(2).transpose(1, 2).contiguous()

    def _decode_boxes(self, raw: Tensor, stride: int) -> Tensor:
        batch, _, height, width = raw.shape
        distances = F.softplus(raw)
        y, x = torch.meshgrid(
            torch.arange(height, device=raw.device, dtype=raw.dtype),
            torch.arange(width, device=raw.device, dtype=raw.dtype),
            indexing="ij",
        )
        center_x = (x + 0.5).reshape(1, 1, height, width)
        center_y = (y + 0.5).reshape(1, 1, height, width)
        left, top, right, bottom = distances.chunk(4, dim=1)
        image_h, image_w = self.input_size
        boxes = torch.cat(
            (
                (center_x - left) * stride / image_w,
                (center_y - top) * stride / image_h,
                (center_x + right) * stride / image_w,
                (center_y + bottom) * stride / image_h,
            ),
            dim=1,
        )
        return self._flatten(boxes.clamp(0.0, 1.0)).reshape(batch, -1, 4)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> dict[str, Tensor]:
        outputs: dict[str, list[Tensor]] = {
            "boxes": [],
            "box_distances": [],
            "objectness_logits": [],
            "stage_logits": [],
            "onset_logits": [],
            "track_embeddings": [],
            "action_logits": [],
        }
        stride_outputs: list[Tensor] = []

        for feature, stride in zip(features, self.strides, strict=True):
            hidden = self.stem(feature)
            raw_boxes = self.box(hidden)
            outputs["boxes"].append(self._decode_boxes(raw_boxes, stride))
            outputs["box_distances"].append(self._flatten(F.softplus(raw_boxes)))
            outputs["objectness_logits"].append(
                self._flatten(self.objectness(hidden)).squeeze(-1)
            )
            outputs["stage_logits"].append(self._flatten(self.stage(hidden)))
            outputs["onset_logits"].append(
                self._flatten(self.onset(hidden)).squeeze(-1)
            )
            embedding = self._flatten(self.embedding(hidden))
            outputs["track_embeddings"].append(
                F.normalize(embedding, p=2.0, dim=-1, eps=1e-6)
            )
            outputs["action_logits"].append(self._flatten(self.action(hidden)))
            stride_outputs.append(
                torch.full(
                    (feature.shape[-2] * feature.shape[-1],),
                    stride,
                    dtype=feature.dtype,
                    device=feature.device,
                )
            )

        result = {
            name: torch.cat(values, dim=1) for name, values in outputs.items()
        }
        result["strides"] = torch.cat(stride_outputs, dim=0)
        return result
