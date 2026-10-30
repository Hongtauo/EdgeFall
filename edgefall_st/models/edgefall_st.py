"""Complete EdgeFall-ST network and streaming feature state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from edgefall_st.models.backbone import LiteBackbone
from edgefall_st.models.cmtf import MultiScaleTemporalFusion
from edgefall_st.models.head import UnifiedAnchorFreeHead
from edgefall_st.models.neck import LitePANFPN

DEFAULT_HISTORY_OFFSETS: dict[str, tuple[int, ...]] = {
    "p3": (0, 1, 2, 4),
    "p4": (0, 1, 2, 4, 8, 16, 24, 31),
    "p5": (0, 1, 2, 4, 8, 16, 24, 31),
}
GLOBAL_FRAME_OFFSETS = tuple(
    sorted({offset for offsets in DEFAULT_HISTORY_OFFSETS.values() for offset in offsets})
)


@dataclass
class StreamState:
    """Runtime-only ring buffer of detached Backbone features."""

    max_history: int
    features: dict[str, deque[Tensor]] = field(
        default_factory=lambda: {
            "p3": deque(),
            "p4": deque(),
            "p5": deque(),
        }
    )

    def reset(self) -> None:
        for values in self.features.values():
            values.clear()

    def sequence(
        self,
        scale: str,
        current: Tensor,
        offsets: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        past = self.features[scale]
        selected: list[Tensor] = []
        valid: list[bool] = []
        for offset in offsets:
            if offset == 0:
                selected.append(current)
                valid.append(True)
            elif len(past) >= offset:
                selected.append(past[-offset])
                valid.append(True)
            else:
                selected.append(torch.zeros_like(current))
                valid.append(False)
        sequence = torch.stack(selected, dim=1)
        mask = torch.tensor(valid, device=current.device, dtype=current.dtype)
        mask = mask.unsqueeze(0).expand(current.shape[0], -1)
        return sequence, mask

    def update(self, current: dict[str, Tensor]) -> None:
        for scale, values in current.items():
            buffer = self.features[scale]
            buffer.append(values.detach())
            while len(buffer) > self.max_history:
                buffer.popleft()


class EdgeFallST(nn.Module):
    """Single learned network for streaming person and fall-stage detection."""

    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (288, 512),
        backbone_channels: tuple[int, int, int, int, int] = (
            16,
            32,
            64,
            128,
            192,
        ),
        neck_channels: int = 96,
        embedding_dim: int = 8,
        num_stages: int = 3,
        num_actions: int = 16,
        history_offsets: dict[str, tuple[int, ...]] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_offsets = history_offsets or DEFAULT_HISTORY_OFFSETS
        self.global_frame_offsets = tuple(
            sorted(
                {
                    offset
                    for offsets in self.history_offsets.values()
                    for offset in offsets
                }
            )
        )
        if self.global_frame_offsets[0] != 0:
            raise ValueError("history offsets must include current frame offset 0")

        self.backbone = LiteBackbone(
            channels=backbone_channels,
            activation=activation,
        )
        self.temporal_fusion = MultiScaleTemporalFusion(
            self.backbone.output_channels,
            self.history_offsets,
            activation=activation,
        )
        self.neck = LitePANFPN(
            self.backbone.output_channels,
            output_channels=neck_channels,
            activation=activation,
        )
        self.head = UnifiedAnchorFreeHead(
            neck_channels,
            input_size=input_size,
            embedding_dim=embedding_dim,
            num_stages=num_stages,
            num_actions=num_actions,
            activation=activation,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def create_stream_state(self) -> StreamState:
        return StreamState(max_history=max(self.global_frame_offsets))

    def _zero_history(
        self,
        current: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        sequences: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for scale, feature in current.items():
            offsets = self.history_offsets[scale]
            slots = [feature, *[torch.zeros_like(feature) for _ in offsets[1:]]]
            sequences[scale] = torch.stack(slots, dim=1)
            mask = feature.new_zeros((feature.shape[0], len(offsets)))
            mask[:, 0] = 1
            masks[scale] = mask
        return sequences, masks

    def _predict(
        self,
        sequences: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        fused = self.temporal_fusion(sequences, masks)
        pyramid = self.neck(fused)
        return self.head(pyramid)

    def forward_frame(self, image: Tensor) -> dict[str, Tensor]:
        """Stateless current-frame inference with empty temporal history."""

        current = self.backbone(image)
        sequences, masks = self._zero_history(current)
        return self._predict(sequences, masks)

    def forward_history(
        self,
        images: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward over sparse current-first history frames.

        The K input slots follow ``self.global_frame_offsets``. With the default
        configuration they represent [t, t-1, t-2, t-4, t-8, t-16, t-24, t-31].
        """

        if images.ndim != 5:
            raise ValueError("images must have shape [B, K, 3, H, W]")
        batch, slots, channels, height, width = images.shape
        if channels != 3 or slots != len(self.global_frame_offsets):
            raise ValueError(
                f"Expected K={len(self.global_frame_offsets)}, C=3; "
                f"received K={slots}, C={channels}"
            )
        if valid_mask is None:
            valid_mask = images.new_ones((batch, slots))
        if valid_mask.shape != (batch, slots):
            raise ValueError("valid_mask must have shape [B, K]")

        flat_features = self.backbone(
            images.reshape(batch * slots, channels, height, width)
        )
        offset_to_index = {
            offset: index for index, offset in enumerate(self.global_frame_offsets)
        }
        sequences: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for scale, values in flat_features.items():
            values = values.reshape(
                batch,
                slots,
                values.shape[1],
                values.shape[2],
                values.shape[3],
            )
            indices = [
                offset_to_index[offset] for offset in self.history_offsets[scale]
            ]
            sequences[scale] = values[:, indices]
            masks[scale] = valid_mask[:, indices]
        return self._predict(sequences, masks)

    def forward_indexed_history(
        self,
        unique_images: Tensor,
        history_indices: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run each distinct source frame once and gather window histories.

        ``unique_images`` contains the frame pool for the complete batch.
        ``history_indices`` maps every [B, K] window slot into that pool and
        uses -1 for padded history positions.
        """

        if unique_images.ndim != 4:
            raise ValueError("unique_images must have shape [U, 3, H, W]")
        unique_count, channels, _, _ = unique_images.shape
        if unique_count == 0 or channels != 3:
            raise ValueError("unique_images must contain at least one RGB frame")
        if history_indices.ndim != 2:
            raise ValueError("history_indices must have shape [B, K]")
        if history_indices.dtype != torch.long:
            raise ValueError("history_indices must use torch.long dtype")
        batch, slots = history_indices.shape
        if slots != len(self.global_frame_offsets):
            raise ValueError(
                f"Expected K={len(self.global_frame_offsets)}; "
                f"received K={slots}"
            )
        if history_indices.device != unique_images.device:
            raise ValueError(
                "history_indices and unique_images must be on the same device"
            )
        if valid_mask is None:
            valid_mask = history_indices.ge(0).to(dtype=unique_images.dtype)
        if valid_mask.shape != (batch, slots):
            raise ValueError("valid_mask must have shape [B, K]")
        if valid_mask.device != unique_images.device:
            raise ValueError(
                "valid_mask and unique_images must be on the same device"
            )

        unique_features = self.backbone(unique_images)
        safe_indices = history_indices.clamp_min(0)
        effective_mask = valid_mask * history_indices.ge(0).to(valid_mask.dtype)
        offset_to_index = {
            offset: index for index, offset in enumerate(self.global_frame_offsets)
        }
        sequences: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for scale, values in unique_features.items():
            gathered = values[safe_indices]
            indices = [
                offset_to_index[offset] for offset in self.history_offsets[scale]
            ]
            sequences[scale] = gathered[:, indices]
            masks[scale] = effective_mask[:, indices]
        return self._predict(sequences, masks)

    def forward_stream(
        self,
        image: Tensor,
        state: StreamState,
    ) -> dict[str, Tensor]:
        """Streaming inference that computes the current Backbone once."""

        current = self.backbone(image)
        sequences: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for scale, feature in current.items():
            sequences[scale], masks[scale] = state.sequence(
                scale,
                feature,
                self.history_offsets[scale],
            )
        outputs = self._predict(sequences, masks)
        state.update(current)
        return outputs

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim == 4:
            return self.forward_frame(images)
        if images.ndim == 5:
            return self.forward_history(images)
        raise ValueError("Expected [B,3,H,W] or [B,K,3,H,W] input")
