"""Causal multi-scale temporal fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from edgefall_st.models.layers import ConvBNAct, InvertedResidual


class CausalTemporalFusion(nn.Module):
    """Fuse fixed current-first temporal feature slots with 2D operators."""

    def __init__(
        self,
        channels: int,
        slots: int,
        *,
        compression_ratio: int = 4,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if slots < 2:
            raise ValueError("Temporal fusion requires at least two slots")
        compressed_channels = max(8, channels // compression_ratio)
        self.channels = channels
        self.slots = slots
        self.compressed_channels = compressed_channels
        self.compress = ConvBNAct(
            channels,
            compressed_channels,
            1,
            activation=activation,
        )
        fusion_channels = compressed_channels * (2 * slots - 1)
        self.fuse = nn.Sequential(
            ConvBNAct(
                fusion_channels,
                channels,
                1,
                activation=activation,
            ),
            InvertedResidual(
                channels,
                channels,
                expansion=1.5,
                activation=activation,
            ),
        )

    def forward(
        self,
        features: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Fuse [B, K, C, H, W], where slot zero is the current frame."""

        if features.ndim != 5:
            raise ValueError("features must have shape [B, K, C, H, W]")
        batch, slots, channels, height, width = features.shape
        if slots != self.slots or channels != self.channels:
            raise ValueError(
                f"Expected K={self.slots}, C={self.channels}; "
                f"received K={slots}, C={channels}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, slots),
                dtype=features.dtype,
                device=features.device,
            )
        if valid_mask.shape != (batch, slots):
            raise ValueError("valid_mask must have shape [B, K]")

        compressed = self.compress(
            features.reshape(batch * slots, channels, height, width)
        )
        compressed = compressed.reshape(
            batch,
            slots,
            self.compressed_channels,
            height,
            width,
        )
        mask = valid_mask.to(compressed.dtype).view(batch, slots, 1, 1, 1)
        compressed = compressed * mask

        current = compressed[:, :1]
        differences = (current - compressed[:, 1:]) * mask[:, 1:]
        fused_inputs = torch.cat(
            [
                compressed.flatten(1, 2),
                differences.flatten(1, 2),
            ],
            dim=1,
        )
        return features[:, 0] + self.fuse(fused_inputs)


class MultiScaleTemporalFusion(nn.Module):
    def __init__(
        self,
        channels: dict[str, int],
        history_offsets: dict[str, tuple[int, ...]],
        *,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.history_offsets = history_offsets
        self.blocks = nn.ModuleDict(
            {
                scale: CausalTemporalFusion(
                    channels[scale],
                    len(history_offsets[scale]),
                    activation=activation,
                )
                for scale in ("p3", "p4", "p5")
            }
        )

    def forward(
        self,
        sequences: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return {
            scale: self.blocks[scale](sequences[scale], masks[scale])
            for scale in ("p3", "p4", "p5")
        }
