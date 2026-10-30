"""BBox-conditioned temporal classifier for fall-event discrimination."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from edgefall_st.data.dataset import DEFAULT_HISTORY_OFFSETS
from edgefall_st.models.backbone import LiteBackbone


V32_SHORT_HISTORY_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7)
V32_LONG_HISTORY_OFFSETS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
V33_VISUAL_SHORT_HISTORY_OFFSETS = (0, 2, 4, 7)
V33_VISUAL_LONG_HISTORY_OFFSETS = (0, 8, 16, 24, 32)
V33_TEMPORAL_MODELS = frozenset({"v33-g", "v33-gr"})


class ActorCropper(nn.Module):
    """Crop one context-preserving actor region from each input frame."""

    def __init__(
        self,
        *,
        output_size: tuple[int, int] = (128, 128),
        context_scale: float = 1.25,
    ) -> None:
        super().__init__()
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output_size dimensions must be positive")
        if context_scale < 1.0:
            raise ValueError("context_scale must be at least 1")
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.context_scale = float(context_scale)

    def _context_boxes(self, boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
        image_h, image_w = image_size
        pixel_scale = boxes.new_tensor(
            [image_w, image_h, image_w, image_h]
        )
        pixel_boxes = boxes.clamp(0.0, 1.0) * pixel_scale
        left, top, right, bottom = pixel_boxes.unbind(dim=-1)
        width = (right - left).clamp_min(1.0) * self.context_scale
        height = (bottom - top).clamp_min(1.0) * self.context_scale

        target_aspect = self.output_size[1] / self.output_size[0]
        expanded_width = torch.maximum(width, height * target_aspect)
        expanded_height = torch.maximum(height, width / target_aspect)
        expanded_width = expanded_width.clamp(max=float(image_w))
        expanded_height = expanded_height.clamp(max=float(image_h))

        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        half_width = expanded_width * 0.5
        half_height = expanded_height * 0.5
        center_x = torch.maximum(
            half_width,
            torch.minimum(
                center_x,
                center_x.new_full(center_x.shape, float(image_w)) - half_width,
            ),
        )
        center_y = torch.maximum(
            half_height,
            torch.minimum(
                center_y,
                center_y.new_full(center_y.shape, float(image_h)) - half_height,
            ),
        )
        return torch.stack(
            (
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
            ),
            dim=-1,
        )

    def forward(self, images: Tensor, boxes: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [N, 3, H, W]")
        if boxes.shape != (images.shape[0], 4):
            raise ValueError("boxes must have shape [N, 4]")
        context_boxes = self._context_boxes(
            boxes.to(dtype=images.dtype),
            (int(images.shape[-2]), int(images.shape[-1])),
        )
        batch_indices = torch.arange(
            images.shape[0],
            device=images.device,
            dtype=context_boxes.dtype,
        )[:, None]
        regions = torch.cat((batch_indices, context_boxes), dim=1)
        return roi_align(
            images,
            regions,
            output_size=self.output_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )


class CausalTemporalBlock(nn.Module):
    """Depthwise causal temporal convolution with a masked residual path."""

    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0 or dilation <= 0 or expansion <= 0:
            raise ValueError("Temporal block dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.left_padding = 2 * int(dilation)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
        )
        hidden_channels = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden_channels)
        self.contract = nn.Linear(hidden_channels, channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, K, C]")
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError("valid_mask must have shape [B, K]")
        mask = valid_mask.to(dtype=tokens.dtype)[:, :, None]
        residual = tokens * mask
        temporal = self.depthwise(
            F.pad(
                residual.transpose(1, 2),
                (self.left_padding, 0),
            )
        ).transpose(1, 2)
        update = self.contract(
            self.dropout(self.activation(self.expand(self.norm(temporal))))
        )
        return (residual + self.dropout(update)) * mask


class BBoxFallClassifier(nn.Module):
    """Classify fall state from labeled actor boxes and sparse RGB history.

    This model deliberately has no box-regression or person-objectness head.
    BBoxes are inputs, so its metrics measure semantic discrimination under
    externally supplied localization.
    """

    def __init__(
        self,
        *,
        input_size: tuple[int, int] = (288, 512),
        crop_size: tuple[int, int] = (128, 128),
        crop_context: float = 1.25,
        backbone_channels: tuple[int, int, int, int, int] = (
            16,
            32,
            64,
            128,
            192,
        ),
        visual_dim: int = 128,
        temporal_dim: int = 128,
        history_offsets: Sequence[int] = DEFAULT_HISTORY_OFFSETS,
        num_actions: int = 16,
        activation: str = "silu",
        temporal_model: str = "dual-tcn",
        temporal_dropout: float = 0.1,
        short_horizon: int = 4,
        spatial_pool_size: int = 1,
        short_history_offsets: Sequence[int] | None = None,
        long_history_offsets: Sequence[int] | None = None,
        effective_fps: float = 10.0,
        pre_cropped_inputs: bool = False,
        visual_modality_dropout: float = 0.3,
        visual_gate_bias: float = -2.0,
        visual_input_size: int | None = None,
    ) -> None:
        super().__init__()
        offsets = tuple(int(value) for value in history_offsets)
        if not offsets or offsets[0] != 0:
            raise ValueError("history_offsets must start with current offset 0")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError("history_offsets must be unique and increasing")
        if visual_dim <= 0 or temporal_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if temporal_model not in {
            "attention",
            "dual-tcn",
            "v32",
            *V33_TEMPORAL_MODELS,
        }:
            raise ValueError(
                "temporal_model must be 'attention', 'dual-tcn', 'v32', "
                "'v33-g', or 'v33-gr'"
            )
        if not 0.0 <= temporal_dropout < 1.0:
            raise ValueError("temporal_dropout must be in [0, 1)")
        if short_horizon <= 0:
            raise ValueError("short_horizon must be positive")
        if spatial_pool_size <= 0:
            raise ValueError("spatial_pool_size must be positive")
        if effective_fps <= 0:
            raise ValueError("effective_fps must be positive")
        if not 0.0 <= visual_modality_dropout <= 1.0:
            raise ValueError("visual_modality_dropout must be in [0, 1]")
        if visual_input_size is not None and visual_input_size <= 0:
            raise ValueError("visual_input_size must be positive")

        short_offsets = tuple(
            int(value)
            for value in (
                V32_SHORT_HISTORY_OFFSETS
                if short_history_offsets is None
                else short_history_offsets
            )
        )
        long_offsets = tuple(
            int(value)
            for value in (
                V32_LONG_HISTORY_OFFSETS
                if long_history_offsets is None
                else long_history_offsets
            )
        )
        if temporal_model == "v32" or temporal_model in V33_TEMPORAL_MODELS:
            for name, branch_offsets in (
                ("short_history_offsets", short_offsets),
                ("long_history_offsets", long_offsets),
            ):
                if not branch_offsets or branch_offsets[0] != 0:
                    raise ValueError(f"{name} must start with 0")
                if tuple(sorted(set(branch_offsets))) != branch_offsets:
                    raise ValueError(
                        f"{name} must be unique and increasing"
                    )
                if not set(branch_offsets).issubset(offsets):
                    raise ValueError(
                        f"{name} must be a subset of history_offsets"
                    )

        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.history_offsets = offsets
        self.temporal_model = temporal_model
        self.short_horizon = int(short_horizon)
        self.spatial_pool_size = int(spatial_pool_size)
        self.short_history_offsets = short_offsets
        self.long_history_offsets = long_offsets
        self.effective_fps = float(effective_fps)
        self.pre_cropped_inputs = bool(pre_cropped_inputs)
        self.visual_modality_dropout = float(visual_modality_dropout)
        self.visual_gate_bias = float(visual_gate_bias)
        self.visual_dim = int(visual_dim)
        self.geometry_frozen = False
        self.visual_input_size = (
            int(visual_input_size)
            if temporal_model == "v33-gr" and visual_input_size is not None
            else None
        )
        index_by_offset = {
            offset: index for index, offset in enumerate(self.history_offsets)
        }
        if temporal_model == "v33-gr":
            self.visual_short_history_offsets = (
                V33_VISUAL_SHORT_HISTORY_OFFSETS
            )
            self.visual_long_history_offsets = V33_VISUAL_LONG_HISTORY_OFFSETS
            for name, visual_offsets in (
                (
                    "V33_VISUAL_SHORT_HISTORY_OFFSETS",
                    self.visual_short_history_offsets,
                ),
                (
                    "V33_VISUAL_LONG_HISTORY_OFFSETS",
                    self.visual_long_history_offsets,
                ),
            ):
                if not set(visual_offsets).issubset(index_by_offset):
                    raise ValueError(f"{name} must be a subset of history_offsets")
            self.visual_history_slot_indices = tuple(
                index_by_offset[offset]
                for offset in sorted(
                    set(self.visual_short_history_offsets)
                    | set(self.visual_long_history_offsets)
                )
            )
            self.visual_short_position_indices = tuple(
                short_offsets.index(offset)
                for offset in self.visual_short_history_offsets
            )
            self.visual_long_position_indices = tuple(
                long_offsets.index(offset)
                for offset in self.visual_long_history_offsets
            )
        else:
            self.visual_short_history_offsets = short_offsets
            self.visual_long_history_offsets = long_offsets
            self.visual_history_slot_indices = tuple(range(len(offsets)))
            self.visual_short_position_indices = tuple(
                range(len(short_offsets))
            )
            self.visual_long_position_indices = tuple(
                range(len(long_offsets))
            )
        self.cropper = ActorCropper(
            output_size=self.crop_size,
            context_scale=crop_context,
        )
        activation_layer: nn.Module
        if activation == "silu":
            activation_layer = nn.SiLU(inplace=True)
        elif activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        if temporal_model == "v33-g":
            self.backbone = nn.Identity()
            self.visual_projection = nn.Identity()
        else:
            self.backbone = LiteBackbone(
                channels=backbone_channels,
                activation=activation,
            )
            pooled_dim = (
                sum(self.backbone.output_channels.values())
                * self.spatial_pool_size**2
            )
            self.visual_projection = nn.Sequential(
                nn.Linear(pooled_dim, visual_dim),
                nn.LayerNorm(visual_dim),
                activation_layer,
            )

        geometry_dim = 7
        if temporal_model == "attention":
            temporal_input_dim = 2 * visual_dim + 2 * geometry_dim
            self.temporal_projection = nn.Sequential(
                nn.Linear(temporal_input_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(len(offsets), temporal_dim)
            )
            nn.init.normal_(self.position_embedding, std=0.02)
            self.current_query = nn.Linear(temporal_dim, temporal_dim)
            self.attention = nn.Linear(temporal_dim, 1)
            self.summary = nn.Sequential(
                nn.Linear(3 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=0.1),
            )
        elif temporal_model == "dual-tcn":
            # Visual appearance and current-relative visual velocity, plus
            # static geometry, three geometry derivatives and three time cues.
            temporal_input_dim = 2 * visual_dim + 4 * geometry_dim + 3
            self.temporal_projection = nn.Sequential(
                nn.Linear(temporal_input_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(len(offsets), temporal_dim)
            )
            nn.init.normal_(self.position_embedding, std=0.02)
            self.long_temporal = nn.ModuleList(
                CausalTemporalBlock(
                    temporal_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in (1, 2, 4)
            )
            self.short_temporal = nn.ModuleList(
                CausalTemporalBlock(
                    temporal_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in (1, 2, 4)
            )
            self.event_summary = nn.Sequential(
                nn.Linear(4 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
            self.onset_summary = nn.Sequential(
                nn.Linear(3 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
        elif temporal_model == "v32":
            # V3.2 keeps regular short/long timelines separate and retains a
            # coarse spatial layout before temporal fusion.
            temporal_input_dim = 2 * visual_dim + 4 * geometry_dim + 3
            self.temporal_projection = nn.Sequential(
                nn.Linear(temporal_input_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
            )
            self.short_position_embedding = nn.Parameter(
                torch.zeros(len(short_offsets), temporal_dim)
            )
            self.long_position_embedding = nn.Parameter(
                torch.zeros(len(long_offsets), temporal_dim)
            )
            nn.init.normal_(self.short_position_embedding, std=0.02)
            nn.init.normal_(self.long_position_embedding, std=0.02)
            self.long_temporal = nn.ModuleList(
                CausalTemporalBlock(
                    temporal_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in (1, 2, 4)
            )
            self.short_temporal = nn.ModuleList(
                CausalTemporalBlock(
                    temporal_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in (1, 2, 4)
            )
            self.posture_summary = nn.Sequential(
                nn.Linear(visual_dim + geometry_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
            self.onset_summary = nn.Sequential(
                nn.Linear(3 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
            self.control_summary = nn.Sequential(
                nn.Linear(3 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
            self.event_summary = nn.Sequential(
                nn.Linear(3 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
        else:
            # V3.3 makes geometry independently predictive. RGB can only add
            # a gated residual in v33-gr; v33-g omits the visual backbone.
            geometry_token_dim = 4 * geometry_dim + 3
            geometry_statistics_dim = 9 * geometry_dim
            self.geometry_projection = nn.Sequential(
                nn.Linear(geometry_token_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
            )
            self.short_position_embedding = nn.Parameter(
                torch.zeros(len(short_offsets), temporal_dim)
            )
            self.long_position_embedding = nn.Parameter(
                torch.zeros(len(long_offsets), temporal_dim)
            )
            nn.init.normal_(self.short_position_embedding, std=0.02)
            nn.init.normal_(self.long_position_embedding, std=0.02)
            self.geometry_short_temporal = nn.ModuleList(
                CausalTemporalBlock(
                    temporal_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in (1, 2, 4)
            )
            self.geometry_long_temporal = nn.ModuleList(
                CausalTemporalBlock(
                    temporal_dim,
                    dilation=dilation,
                    dropout=temporal_dropout,
                )
                for dilation in (1, 2, 4)
            )
            self.geometry_summary = nn.Sequential(
                nn.Linear(
                    3 * temporal_dim + 2 * geometry_statistics_dim,
                    temporal_dim,
                ),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
            self.geometry_onset_summary = nn.Sequential(
                nn.Linear(3 * temporal_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=temporal_dropout),
            )
            if temporal_model == "v33-gr":
                visual_token_dim = 2 * visual_dim + 3
                self.visual_temporal_projection = nn.Sequential(
                    nn.Linear(visual_token_dim, temporal_dim),
                    nn.LayerNorm(temporal_dim),
                    nn.SiLU(inplace=True),
                )
                self.visual_short_temporal = nn.ModuleList(
                    CausalTemporalBlock(
                        temporal_dim,
                        dilation=dilation,
                        dropout=temporal_dropout,
                    )
                    for dilation in (1, 2, 4)
                )
                self.visual_long_temporal = nn.ModuleList(
                    CausalTemporalBlock(
                        temporal_dim,
                        dilation=dilation,
                        dropout=temporal_dropout,
                    )
                    for dilation in (1, 2, 4)
                )
                self.visual_summary = nn.Sequential(
                    nn.Linear(3 * temporal_dim, temporal_dim),
                    nn.LayerNorm(temporal_dim),
                    nn.SiLU(inplace=True),
                    nn.Dropout(p=temporal_dropout),
                )
                self.visual_gate = nn.Linear(temporal_dim, 1)
                nn.init.constant_(self.visual_gate.bias, self.visual_gate_bias)
                self.visual_event_residual = nn.Linear(temporal_dim, 2)
                self.visual_stage_residual = nn.Linear(temporal_dim, 3)
                self.visual_onset_residual = nn.Linear(temporal_dim, 1)
                self.visual_action_residual = nn.Linear(
                    temporal_dim,
                    num_actions,
                )
                for residual_head in (
                    self.visual_event_residual,
                    self.visual_stage_residual,
                    self.visual_onset_residual,
                    self.visual_action_residual,
                ):
                    nn.init.zeros_(residual_head.weight)
                    nn.init.zeros_(residual_head.bias)
        self.stage_head = nn.Linear(temporal_dim, 3)
        self.fall_head = (
            None
            if temporal_model in V33_TEMPORAL_MODELS
            else nn.Linear(temporal_dim, 1)
        )
        self.onset_head = nn.Linear(temporal_dim, 1)
        self.action_head = nn.Linear(temporal_dim, num_actions)
        self.event_state_head = (
            nn.Linear(temporal_dim, 2)
            if temporal_model in V33_TEMPORAL_MODELS
            else None
        )
        self.control_head = (
            nn.Linear(temporal_dim, 1)
            if temporal_model == "v32" or temporal_model in V33_TEMPORAL_MODELS
            else None
        )
        prior_logit = math.log(0.05 / 0.95)
        if self.fall_head is not None:
            nn.init.constant_(self.fall_head.bias, prior_logit)
        else:
            if self.event_state_head is None:
                raise RuntimeError("V3.3 event-state head is not initialized")
            state_probability = 1.0 - math.sqrt(0.95)
            state_prior_logit = math.log(
                state_probability / (1.0 - state_probability)
            )
            nn.init.constant_(self.event_state_head.bias, state_prior_logit)
            nn.init.zeros_(self.stage_head.bias)
            nn.init.constant_(self.stage_head.bias[1:], prior_logit)
        nn.init.constant_(self.onset_head.bias, prior_logit)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _is_visual_parameter(name: str) -> bool:
        return name.startswith(
            (
                "backbone.",
                "visual_projection.",
                "visual_temporal_projection.",
                "visual_short_temporal.",
                "visual_long_temporal.",
                "visual_summary.",
                "visual_gate.",
                "visual_event_residual.",
                "visual_stage_residual.",
                "visual_onset_residual.",
                "visual_action_residual.",
            )
        )

    def load_geometry_state_dict(self, state_dict: dict[str, Tensor]) -> int:
        if self.temporal_model != "v33-gr":
            raise ValueError("Geometry initialization requires v33-gr")
        current = self.state_dict()
        expected = {
            name
            for name in current
            if not self._is_visual_parameter(name)
        }
        compatible = {
            name: value
            for name, value in state_dict.items()
            if name in expected and current[name].shape == value.shape
        }
        missing = sorted(expected - compatible.keys())
        if missing:
            raise ValueError(
                "Geometry checkpoint is missing compatible parameters: "
                + ", ".join(missing[:5])
            )
        self.load_state_dict(compatible, strict=False)
        return len(compatible)

    def freeze_geometry_branch(self) -> None:
        if self.temporal_model != "v33-gr":
            raise ValueError("Geometry freezing requires v33-gr")
        for name, parameter in self.named_parameters():
            if not self._is_visual_parameter(name):
                parameter.requires_grad_(False)
        self.geometry_frozen = True
        self._set_frozen_geometry_eval()

    def _set_frozen_geometry_eval(self) -> None:
        for name in (
            "geometry_projection",
            "geometry_short_temporal",
            "geometry_long_temporal",
            "geometry_summary",
            "geometry_onset_summary",
            "stage_head",
            "onset_head",
            "action_head",
            "event_state_head",
            "control_head",
        ):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.eval()

    def train(self, mode: bool = True) -> BBoxFallClassifier:
        super().train(mode)
        if mode and self.geometry_frozen:
            self._set_frozen_geometry_eval()
        return self

    @staticmethod
    def _geometry(boxes: Tensor) -> Tensor:
        left, top, right, bottom = boxes.unbind(dim=-1)
        width = (right - left).clamp_min(1e-4)
        height = (bottom - top).clamp_min(1e-4)
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        area = width * height
        log_aspect = torch.log(width / height).clamp(-5.0, 5.0)
        return torch.stack(
            (center_x, center_y, width, height, area, log_aspect, bottom),
            dim=-1,
        )

    def _encode_images(self, images: Tensor, boxes: Tensor) -> Tensor:
        if self.pre_cropped_inputs:
            if images.shape[-2:] != self.crop_size:
                raise ValueError(
                    "Pre-cropped images must match the configured crop_size"
                )
            crops = images
        else:
            crops = self.cropper(images, boxes)
        if self.visual_input_size is not None and crops.shape[-2:] != (
            self.visual_input_size,
            self.visual_input_size,
        ):
            crops = F.interpolate(
                crops,
                size=(self.visual_input_size, self.visual_input_size),
                mode="area",
            )
        pyramid = self.backbone(crops)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(
                    pyramid[scale],
                    self.spatial_pool_size,
                ).flatten(1)
                for scale in ("p3", "p4", "p5")
            ],
            dim=-1,
        )
        return self.visual_projection(pooled)

    def _classify(
        self,
        visual: Tensor | None,
        boxes: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch, slots, _ = boxes.shape
        if slots != len(self.history_offsets):
            raise ValueError(
                f"Expected K={len(self.history_offsets)}; received K={slots}"
            )
        if boxes.shape != (batch, slots, 4):
            raise ValueError("boxes must have shape [B, K, 4]")
        if valid_mask.shape != (batch, slots):
            raise ValueError("valid_mask must have shape [B, K]")
        if visual is not None and visual.shape[:2] != (batch, slots):
            raise ValueError("visual must have shape [B, K, C]")

        if self.temporal_model == "attention":
            if visual is None:
                raise ValueError("attention temporal model requires visual input")
            return self._classify_attention(visual, boxes, valid_mask)
        if self.temporal_model == "dual-tcn":
            if visual is None:
                raise ValueError("dual-tcn temporal model requires visual input")
            return self._classify_dual_tcn(visual, boxes, valid_mask)
        if self.temporal_model == "v32":
            if visual is None:
                raise ValueError("v32 temporal model requires visual input")
            return self._classify_v32(visual, boxes, valid_mask)
        if self.temporal_model == "v33-gr" and visual is None:
            raise ValueError("v33-gr temporal model requires visual input")
        return self._classify_v33(visual, boxes, valid_mask)

    def _classify_attention(
        self,
        visual: Tensor,
        boxes: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        geometry = self._geometry(boxes)
        visual_delta = visual[:, :1] - visual
        geometry_delta = geometry[:, :1] - geometry
        tokens = self.temporal_projection(
            torch.cat(
                (visual, visual_delta, geometry, geometry_delta),
                dim=-1,
            )
        )
        tokens = tokens + self.position_embedding[None].to(tokens.dtype)
        mask = valid_mask.to(dtype=tokens.dtype)
        tokens = tokens * mask[:, :, None]

        query = self.current_query(tokens[:, :1])
        attention_logits = self.attention(torch.tanh(tokens + query)).squeeze(-1)
        attention_logits = attention_logits.masked_fill(mask <= 0, -1e4)
        attention_weights = attention_logits.softmax(dim=1) * mask
        attention_weights = attention_weights / attention_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        context = (tokens * attention_weights[:, :, None]).sum(dim=1)
        current = tokens[:, 0]
        summary = self.summary(
            torch.cat((current, context, current - context), dim=-1)
        )
        return {
            "stage_logits": self.stage_head(summary),
            "fall_logits": self.fall_head(summary).squeeze(-1),
            "onset_logits": self.onset_head(summary).squeeze(-1),
            "action_logits": self.action_head(summary),
            "attention_weights": attention_weights,
        }

    def _temporal_features(
        self,
        visual: Tensor,
        geometry: Tensor,
        valid_mask: Tensor,
        offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Build chronological tokens with explicit velocity and acceleration."""

        mask = valid_mask.to(dtype=visual.dtype)
        offset_tensor = visual.new_tensor(tuple(offsets))
        elapsed_s = offset_tensor / self.effective_fps
        safe_elapsed_s = elapsed_s.clamp_min(1.0 / self.effective_fps)

        visual_rate = (
            (visual[:, :1] - visual) / safe_elapsed_s[None, :, None]
        ).clamp(-20.0, 20.0)
        geometry_rate = (
            (geometry[:, :1] - geometry)
            / safe_elapsed_s[None, :, None]
        ).clamp(-20.0, 20.0)
        visual_rate[:, 0] = 0.0
        geometry_rate[:, 0] = 0.0

        chronological_visual = visual.flip(1)
        chronological_visual_rate = visual_rate.flip(1)
        chronological_geometry = geometry.flip(1)
        chronological_geometry_rate = geometry_rate.flip(1)
        chronological_mask = mask.flip(1)

        timestamps_s = (-elapsed_s).flip(0)
        delta_s = torch.zeros_like(timestamps_s)
        if timestamps_s.numel() > 1:
            delta_s[1:] = timestamps_s[1:] - timestamps_s[:-1]
        safe_delta_s = delta_s.clamp_min(1.0 / self.effective_fps)

        local_velocity = torch.zeros_like(chronological_geometry)
        pair_mask = torch.zeros_like(chronological_mask)
        if chronological_geometry.shape[1] > 1:
            pair_mask[:, 1:] = (
                chronological_mask[:, 1:]
                * chronological_mask[:, :-1]
            )
            local_velocity[:, 1:] = (
                chronological_geometry[:, 1:]
                - chronological_geometry[:, :-1]
            ) / safe_delta_s[None, 1:, None]
            local_velocity = local_velocity.clamp(-20.0, 20.0)
            local_velocity = local_velocity * pair_mask[:, :, None]

        local_acceleration = torch.zeros_like(chronological_geometry)
        if chronological_geometry.shape[1] > 2:
            triple_mask = pair_mask[:, 2:] * pair_mask[:, 1:-1]
            local_acceleration[:, 2:] = (
                local_velocity[:, 2:] - local_velocity[:, 1:-1]
            ) / safe_delta_s[None, 2:, None]
            local_acceleration = local_acceleration.clamp(-40.0, 40.0)
            local_acceleration[:, 2:] *= triple_mask[:, :, None]

        maximum_elapsed = elapsed_s.max().clamp_min(
            1.0 / self.effective_fps
        )
        normalized_age = (timestamps_s / maximum_elapsed)[None, :, None]
        normalized_age = normalized_age.expand(visual.shape[0], -1, -1)
        normalized_delta = (delta_s / maximum_elapsed)[None, :, None]
        normalized_delta = normalized_delta.expand(visual.shape[0], -1, -1)
        history_fraction = mask.mean(dim=1, keepdim=True)[:, :, None]
        history_fraction = history_fraction.expand(-1, visual.shape[1], -1)

        features = torch.cat(
            (
                chronological_visual,
                chronological_visual_rate,
                chronological_geometry,
                chronological_geometry_rate,
                local_velocity,
                local_acceleration,
                normalized_age,
                normalized_delta,
                history_fraction,
            ),
            dim=-1,
        )
        return features, chronological_mask

    def _dual_tcn_features(
        self,
        visual: Tensor,
        geometry: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features, chronological_mask = self._temporal_features(
            visual,
            geometry,
            valid_mask,
            self.history_offsets,
        )
        offsets = visual.new_tensor(self.history_offsets)
        short_selection = (
            offsets.le(self.short_horizon)
            .to(dtype=visual.dtype)
            .flip(0)
        )
        short_mask = chronological_mask * short_selection[None]
        return features, chronological_mask, short_mask

    def _v32_branch_features(
        self,
        visual: Tensor,
        geometry: Tensor,
        valid_mask: Tensor,
        offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        index_by_offset = {
            offset: index for index, offset in enumerate(self.history_offsets)
        }
        indices = visual.new_tensor(
            [index_by_offset[offset] for offset in offsets],
            dtype=torch.long,
        )
        return self._temporal_features(
            visual.index_select(1, indices),
            geometry.index_select(1, indices),
            valid_mask.index_select(1, indices),
            offsets,
        )

    def _classify_dual_tcn(
        self,
        visual: Tensor,
        boxes: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        geometry = self._geometry(boxes)
        features, chronological_mask, short_mask = self._dual_tcn_features(
            visual,
            geometry,
            valid_mask,
        )
        tokens = self.temporal_projection(features)
        position = self.position_embedding.flip(0)[None].to(tokens.dtype)
        tokens = (tokens + position) * chronological_mask[:, :, None]

        long_tokens = tokens
        for block in self.long_temporal:
            long_tokens = block(long_tokens, chronological_mask)
        short_tokens = tokens * short_mask[:, :, None]
        for block in self.short_temporal:
            short_tokens = block(short_tokens, short_mask)

        current = tokens[:, -1]
        long_state = long_tokens[:, -1]
        short_state = short_tokens[:, -1]
        event_summary = self.event_summary(
            torch.cat(
                (
                    current,
                    short_state,
                    long_state,
                    short_state - long_state,
                ),
                dim=-1,
            )
        )
        onset_summary = self.onset_summary(
            torch.cat(
                (current, short_state, short_state - current),
                dim=-1,
            )
        )
        diagnostic_weights = valid_mask.to(dtype=visual.dtype)
        diagnostic_weights = diagnostic_weights / diagnostic_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        return {
            "stage_logits": self.stage_head(event_summary),
            "fall_logits": self.fall_head(event_summary).squeeze(-1),
            "onset_logits": self.onset_head(onset_summary).squeeze(-1),
            "action_logits": self.action_head(event_summary),
            "attention_weights": diagnostic_weights,
        }

    def _select_temporal_branch(
        self,
        values: Tensor,
        valid_mask: Tensor,
        offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        index_by_offset = {
            offset: index for index, offset in enumerate(self.history_offsets)
        }
        indices = values.new_tensor(
            [index_by_offset[offset] for offset in offsets],
            dtype=torch.long,
        )
        return (
            values.index_select(1, indices),
            valid_mask.index_select(1, indices),
        )

    def _v33_geometry_branch_features(
        self,
        geometry: Tensor,
        valid_mask: Tensor,
        offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        selected, selected_mask = self._select_temporal_branch(
            geometry,
            valid_mask,
            offsets,
        )
        empty_visual = selected.new_empty((*selected.shape[:2], 0))
        features, chronological_mask = self._temporal_features(
            empty_visual,
            selected,
            selected_mask,
            offsets,
        )
        chronological_geometry = selected.flip(1)
        statistics = self._masked_geometry_statistics(
            chronological_geometry,
            chronological_mask,
            offsets,
        )
        return features, chronological_mask, statistics

    def _v33_visual_branch_features(
        self,
        visual: Tensor,
        valid_mask: Tensor,
        offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        selected, selected_mask = self._select_temporal_branch(
            visual,
            valid_mask,
            offsets,
        )
        empty_geometry = selected.new_empty((*selected.shape[:2], 0))
        return self._temporal_features(
            selected,
            empty_geometry,
            selected_mask,
            offsets,
        )

    @staticmethod
    def _merge_temporal_branches(
        branches: Sequence[tuple[Tensor, Tensor, Sequence[int]]],
        output_offsets: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Merge chronological short/long states onto one offset timeline."""

        if not branches:
            raise ValueError("At least one temporal branch is required")
        lookup = [
            {
                int(offset): index
                for index, offset in enumerate(reversed(tuple(offsets)))
            }
            for _, _, offsets in branches
        ]
        merged_tokens: list[Tensor] = []
        merged_masks: list[Tensor] = []
        for offset in reversed(tuple(int(value) for value in output_offsets)):
            values: list[Tensor] = []
            masks: list[Tensor] = []
            for (tokens, valid_mask, _), indices in zip(
                branches, lookup, strict=True
            ):
                index = indices.get(offset)
                if index is None:
                    continue
                values.append(tokens[:, index])
                masks.append(valid_mask[:, index])
            if not values:
                raise ValueError(f"Temporal offset {offset} is not represented")
            valid = torch.stack(masks, dim=0).amax(dim=0)
            value = torch.stack(values, dim=0).mean(dim=0)
            merged_tokens.append(value * valid[:, None].to(value.dtype))
            merged_masks.append(valid)
        return torch.stack(merged_tokens, dim=1), torch.stack(
            merged_masks, dim=1
        )

    def _masked_geometry_statistics(
        self,
        chronological_geometry: Tensor,
        chronological_mask: Tensor,
        offsets: Sequence[int],
    ) -> Tensor:
        """Return compact, normalized trajectory statistics per box signal."""

        values = chronological_geometry
        mask = chronological_mask.to(dtype=values.dtype)
        expanded_mask = mask[:, :, None]
        count = expanded_mask.sum(dim=1).clamp_min(1.0)
        mean = (values * expanded_mask).sum(dim=1) / count
        centered = (values - mean[:, None]) * expanded_mask
        standard_deviation = (
            centered.square().sum(dim=1) / count
        ).sqrt()

        positive_infinity = torch.finfo(values.dtype).max
        minimum = values.masked_fill(
            expanded_mask <= 0,
            positive_infinity,
        ).amin(dim=1)
        maximum = values.masked_fill(
            expanded_mask <= 0,
            -positive_infinity,
        ).amax(dim=1)
        has_valid = mask.any(dim=1)[:, None]
        minimum = torch.where(has_valid, minimum, torch.zeros_like(minimum))
        maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))
        value_range = maximum - minimum

        elapsed_s = values.new_tensor(tuple(offsets)) / self.effective_fps
        maximum_elapsed = elapsed_s.max().clamp_min(
            1.0 / self.effective_fps
        )
        normalized_time = (-elapsed_s).flip(0) / maximum_elapsed
        time = normalized_time[None, :, None]
        mean_time = (time * expanded_mask).sum(dim=1) / count
        centered_time = (time - mean_time[:, None]) * expanded_mask
        slope = (centered_time * centered).sum(dim=1) / (
            centered_time.square().sum(dim=1).clamp_min(1e-6)
        )

        first_index = mask.gt(0).to(torch.long).argmax(dim=1)
        last_index = mask.shape[1] - 1 - mask.flip(1).gt(0).to(
            torch.long
        ).argmax(dim=1)
        gather_shape = (-1, 1, values.shape[-1])
        first = values.gather(
            1,
            first_index[:, None, None].expand(*gather_shape),
        ).squeeze(1)
        last = values.gather(
            1,
            last_index[:, None, None].expand(*gather_shape),
        ).squeeze(1)
        endpoint_delta = torch.where(
            has_valid,
            last - first,
            torch.zeros_like(last),
        )

        velocity = torch.zeros_like(values)
        pair_mask = torch.zeros_like(mask)
        if values.shape[1] > 1:
            chronological_time_s = (-elapsed_s).flip(0)
            delta_s = (
                chronological_time_s[1:] - chronological_time_s[:-1]
            ).clamp_min(1.0 / self.effective_fps)
            pair_mask[:, 1:] = mask[:, 1:] * mask[:, :-1]
            velocity[:, 1:] = (
                values[:, 1:] - values[:, :-1]
            ) / delta_s[None, :, None]
            velocity *= pair_mask[:, :, None]
        absolute_velocity = velocity.abs()
        maximum_velocity, peak_index = absolute_velocity.max(dim=1)
        peak_time = normalized_time[peak_index]
        has_pair = pair_mask.any(dim=1)[:, None]
        maximum_velocity = torch.where(
            has_pair,
            maximum_velocity,
            torch.zeros_like(maximum_velocity),
        )
        peak_time = torch.where(
            has_pair,
            peak_time,
            torch.zeros_like(peak_time),
        )
        return torch.cat(
            (
                mean,
                standard_deviation,
                minimum,
                maximum,
                value_range,
                slope,
                endpoint_delta,
                maximum_velocity,
                peak_time,
            ),
            dim=-1,
        )

    @staticmethod
    def _event_logit(event_state_logits: Tensor) -> Tensor:
        if event_state_logits.shape[-1] != 2:
            raise ValueError("event_state_logits must end with two states")
        first, second = event_state_logits.unbind(dim=-1)
        return torch.logsumexp(
            torch.stack((first, second, first + second), dim=-1),
            dim=-1,
        )

    def _classify_v33(
        self,
        visual: Tensor | None,
        boxes: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        geometry = self._geometry(boxes)
        short_features, short_mask, short_statistics = (
            self._v33_geometry_branch_features(
                geometry,
                valid_mask,
                self.short_history_offsets,
            )
        )
        long_features, long_mask, long_statistics = (
            self._v33_geometry_branch_features(
                geometry,
                valid_mask,
                self.long_history_offsets,
            )
        )
        short_tokens = self.geometry_projection(short_features)
        short_tokens = (
            short_tokens
            + self.short_position_embedding[None].to(short_tokens.dtype)
        ) * short_mask[:, :, None]
        long_tokens = self.geometry_projection(long_features)
        long_tokens = (
            long_tokens
            + self.long_position_embedding[None].to(long_tokens.dtype)
        ) * long_mask[:, :, None]

        short_state_tokens = short_tokens
        for block in self.geometry_short_temporal:
            short_state_tokens = block(short_state_tokens, short_mask)
        long_state_tokens = long_tokens
        for block in self.geometry_long_temporal:
            long_state_tokens = block(long_state_tokens, long_mask)
        geometry_tokens, geometry_token_mask = self._merge_temporal_branches(
            (
                (
                    short_state_tokens,
                    short_mask,
                    self.short_history_offsets,
                ),
                (
                    long_state_tokens,
                    long_mask,
                    self.long_history_offsets,
                ),
            ),
            self.history_offsets,
        )
        short_current = short_tokens[:, -1]
        short_state = short_state_tokens[:, -1]
        long_state = long_state_tokens[:, -1]
        geometry_state = self.geometry_summary(
            torch.cat(
                (
                    short_current,
                    short_state,
                    long_state,
                    short_statistics,
                    long_statistics,
                ),
                dim=-1,
            )
        )
        geometry_onset_state = self.geometry_onset_summary(
            torch.cat(
                (short_current, short_state, short_state - short_current),
                dim=-1,
            )
        )
        if self.event_state_head is None or self.control_head is None:
            raise RuntimeError("V3.3 prediction heads are not initialized")
        geometry_event_logits = self.event_state_head(geometry_state)
        geometry_stage_logits = self.stage_head(geometry_state)
        geometry_onset_logits = self.onset_head(geometry_onset_state).squeeze(-1)
        geometry_action_logits = self.action_head(geometry_state)
        geometry_control_logits = self.control_head(geometry_state).squeeze(-1)

        event_state_logits = geometry_event_logits
        stage_logits = geometry_stage_logits
        onset_logits = geometry_onset_logits
        action_logits = geometry_action_logits
        visual_gate = geometry_state.new_zeros(geometry_state.shape[0])
        visual_event_residual = geometry_event_logits.new_zeros(
            geometry_event_logits.shape
        )
        if self.temporal_model == "v33-gr":
            if visual is None:
                raise ValueError("v33-gr requires visual input")
            visual_short_features, visual_short_mask = (
                self._v33_visual_branch_features(
                    visual,
                    valid_mask,
                    self.visual_short_history_offsets,
                )
            )
            visual_long_features, visual_long_mask = (
                self._v33_visual_branch_features(
                    visual,
                    valid_mask,
                    self.visual_long_history_offsets,
                )
            )
            visual_short_tokens = self.visual_temporal_projection(
                visual_short_features
            )
            visual_short_tokens = (
                visual_short_tokens
                + self.short_position_embedding.index_select(
                    0,
                    visual_short_tokens.new_tensor(
                        self.visual_short_position_indices,
                        dtype=torch.long,
                    ),
                )[None].to(visual_short_tokens.dtype)
            ) * visual_short_mask[:, :, None]
            visual_long_tokens = self.visual_temporal_projection(
                visual_long_features
            )
            visual_long_tokens = (
                visual_long_tokens
                + self.long_position_embedding.index_select(
                    0,
                    visual_long_tokens.new_tensor(
                        self.visual_long_position_indices,
                        dtype=torch.long,
                    ),
                )[None].to(visual_long_tokens.dtype)
            ) * visual_long_mask[:, :, None]
            visual_short_state = visual_short_tokens
            for block in self.visual_short_temporal:
                visual_short_state = block(
                    visual_short_state,
                    visual_short_mask,
                )
            visual_long_state = visual_long_tokens
            for block in self.visual_long_temporal:
                visual_long_state = block(
                    visual_long_state,
                    visual_long_mask,
                )
            visual_state = self.visual_summary(
                torch.cat(
                    (
                        visual_short_state[:, -1],
                        visual_long_state[:, -1],
                        visual_short_state[:, -1]
                        - visual_long_state[:, -1],
                    ),
                    dim=-1,
                )
            )
            geometry_event_probability = self._event_logit(
                geometry_event_logits
            ).sigmoid()
            uncertainty = 1.0 - (
                2.0 * geometry_event_probability - 1.0
            ).abs()
            learned_gate = self.visual_gate(
                geometry_state.detach()
            ).squeeze(-1).sigmoid()
            visual_gate = uncertainty.detach() * learned_gate
            if self.training and self.visual_modality_dropout > 0.0:
                keep_visual = torch.rand_like(visual_gate).ge(
                    self.visual_modality_dropout
                )
                visual_gate = visual_gate * keep_visual.to(visual_gate.dtype)

            visual_event_residual = self.visual_event_residual(visual_state)
            event_state_logits = geometry_event_logits + (
                visual_gate[:, None] * visual_event_residual
            )
            stage_logits = geometry_stage_logits + (
                visual_gate[:, None]
                * self.visual_stage_residual(visual_state)
            )
            onset_logits = geometry_onset_logits + (
                visual_gate
                * self.visual_onset_residual(visual_state).squeeze(-1)
            )
            action_logits = geometry_action_logits + (
                visual_gate[:, None]
                * self.visual_action_residual(visual_state)
            )

        diagnostic_weights = valid_mask.to(dtype=boxes.dtype)
        diagnostic_weights = diagnostic_weights / diagnostic_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        return {
            "geometry_state": geometry_state,
            "geometry_tokens": geometry_tokens,
            "geometry_token_mask": geometry_token_mask,
            "stage_logits": stage_logits,
            "fall_logits": self._event_logit(event_state_logits),
            "falling_logits": event_state_logits[:, 0],
            "fallen_logits": event_state_logits[:, 1],
            "onset_logits": onset_logits,
            "action_logits": action_logits,
            "control_logits": geometry_control_logits,
            "geometry_stage_logits": geometry_stage_logits,
            "geometry_fall_logits": self._event_logit(
                geometry_event_logits
            ),
            "geometry_falling_logits": geometry_event_logits[:, 0],
            "geometry_fallen_logits": geometry_event_logits[:, 1],
            "visual_gate": visual_gate,
            "visual_event_residual": visual_event_residual,
            "attention_weights": diagnostic_weights,
        }

    def forward_geometry_history(
        self,
        actor_boxes: Tensor,
        valid_mask: Tensor,
        actor_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Run a V3.3-G geometry branch without constructing dummy images."""

        if self.temporal_model != "v33-g":
            raise ValueError("forward_geometry_history requires v33-g")
        if actor_boxes.ndim != 3 or actor_boxes.shape[-1] != 4:
            raise ValueError("actor_boxes must have shape [B, K, 4]")
        if valid_mask.shape != actor_boxes.shape[:2]:
            raise ValueError("valid_mask must have shape [B, K]")
        if actor_valid_mask.shape != actor_boxes.shape[:2]:
            raise ValueError("actor_valid_mask must have shape [B, K]")
        return self._classify(
            None,
            actor_boxes,
            valid_mask * actor_valid_mask,
        )

    def _classify_v32(
        self,
        visual: Tensor,
        boxes: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        geometry = self._geometry(boxes)
        short_features, short_mask = self._v32_branch_features(
            visual,
            geometry,
            valid_mask,
            self.short_history_offsets,
        )
        long_features, long_mask = self._v32_branch_features(
            visual,
            geometry,
            valid_mask,
            self.long_history_offsets,
        )

        short_tokens = self.temporal_projection(short_features)
        short_tokens = (
            short_tokens
            + self.short_position_embedding[None].to(short_tokens.dtype)
        ) * short_mask[:, :, None]
        long_tokens = self.temporal_projection(long_features)
        long_tokens = (
            long_tokens
            + self.long_position_embedding[None].to(long_tokens.dtype)
        ) * long_mask[:, :, None]

        short_state_tokens = short_tokens
        for block in self.short_temporal:
            short_state_tokens = block(short_state_tokens, short_mask)
        long_state_tokens = long_tokens
        for block in self.long_temporal:
            long_state_tokens = block(long_state_tokens, long_mask)

        short_current = short_tokens[:, -1]
        short_state = short_state_tokens[:, -1]
        long_state = long_state_tokens[:, -1]
        current_valid = valid_mask[:, :1].to(dtype=visual.dtype)
        posture_state = self.posture_summary(
            torch.cat((visual[:, 0], geometry[:, 0]), dim=-1)
        ) * current_valid
        motion_state = self.onset_summary(
            torch.cat(
                (
                    short_current,
                    short_state,
                    short_state - short_current,
                ),
                dim=-1,
            )
        )
        control_state = self.control_summary(
            torch.cat(
                (
                    short_state,
                    long_state,
                    short_state - long_state,
                ),
                dim=-1,
            )
        )
        event_state = self.event_summary(
            torch.cat(
                (posture_state, motion_state, control_state),
                dim=-1,
            )
        )
        if self.control_head is None:
            raise RuntimeError("V3.2 control head is not initialized")

        diagnostic_weights = valid_mask.to(dtype=visual.dtype)
        diagnostic_weights = diagnostic_weights / diagnostic_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        return {
            "stage_logits": self.stage_head(event_state),
            "fall_logits": self.fall_head(event_state).squeeze(-1),
            "onset_logits": self.onset_head(motion_state).squeeze(-1),
            "action_logits": self.action_head(event_state),
            "control_logits": self.control_head(control_state).squeeze(-1),
            "attention_weights": diagnostic_weights,
        }

    def forward_history(
        self,
        images: Tensor,
        actor_boxes: Tensor,
        valid_mask: Tensor,
        actor_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, K, 3, H, W]")
        batch, slots, channels, height, width = images.shape
        if channels != 3 or slots != len(self.history_offsets):
            raise ValueError("Unexpected history image shape")
        if actor_boxes.shape != (batch, slots, 4):
            raise ValueError("actor_boxes must have shape [B, K, 4]")
        if actor_valid_mask.shape != (batch, slots):
            raise ValueError("actor_valid_mask must have shape [B, K]")

        effective_mask = valid_mask * actor_valid_mask
        if self.temporal_model == "v33-g":
            return self._classify(None, actor_boxes, effective_mask)
        if self.temporal_model == "v33-gr":
            visual_slots = images.new_tensor(
                self.visual_history_slot_indices,
                dtype=torch.long,
            )
            selected_images = images.index_select(1, visual_slots)
            selected_boxes = actor_boxes.index_select(1, visual_slots)
            selected_count = selected_images.shape[1]
            selected_visual = self._encode_images(
                selected_images.reshape(
                    batch * selected_count,
                    channels,
                    height,
                    width,
                ),
                selected_boxes.reshape(batch * selected_count, 4),
            ).reshape(batch, selected_count, -1)
            visual = selected_visual.new_zeros(
                (batch, slots, selected_visual.shape[-1])
            ).index_copy(1, visual_slots, selected_visual)
        else:
            flat_visual = self._encode_images(
                images.reshape(batch * slots, channels, height, width),
                actor_boxes.reshape(batch * slots, 4),
            )
            visual = flat_visual.reshape(batch, slots, -1)
        return self._classify(visual, actor_boxes, effective_mask)

    def forward_indexed_history(
        self,
        unique_images: Tensor,
        unique_actor_boxes: Tensor,
        history_indices: Tensor,
        valid_mask: Tensor,
        unique_actor_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if unique_images.ndim != 4:
            raise ValueError("unique_images must have shape [U, 3, H, W]")
        if unique_actor_boxes.shape != (unique_images.shape[0], 4):
            raise ValueError("unique_actor_boxes must have shape [U, 4]")
        if unique_actor_valid_mask.shape != (unique_images.shape[0],):
            raise ValueError("unique_actor_valid_mask must have shape [U]")
        if history_indices.ndim != 2:
            raise ValueError("history_indices must have shape [B, K]")
        if history_indices.shape[1] != len(self.history_offsets):
            raise ValueError("history_indices has an unexpected slot count")
        if valid_mask.shape != history_indices.shape:
            raise ValueError("valid_mask and history_indices must have equal shape")

        safe_indices = history_indices.clamp_min(0)
        boxes = unique_actor_boxes[safe_indices]
        actor_mask = unique_actor_valid_mask[safe_indices]
        effective_mask = (
            valid_mask
            * history_indices.ge(0).to(valid_mask.dtype)
            * actor_mask.to(valid_mask.dtype)
        )
        if self.temporal_model == "v33-g":
            return self._classify(None, boxes, effective_mask)
        if self.temporal_model == "v33-gr":
            visual_slots = history_indices.new_tensor(
                self.visual_history_slot_indices
            )
            visual_history_indices = history_indices.index_select(
                1,
                visual_slots,
            )
            referenced_indices = torch.unique(
                visual_history_indices[visual_history_indices.ge(0)]
            )
            if referenced_indices.numel() == 0:
                unique_visual = unique_images.new_zeros(
                    (unique_images.shape[0], self.visual_dim)
                )
            else:
                encoded_visual = self._encode_images(
                    unique_images.index_select(0, referenced_indices),
                    unique_actor_boxes.index_select(0, referenced_indices),
                )
                unique_visual = encoded_visual.new_zeros(
                    (unique_images.shape[0], encoded_visual.shape[-1])
                ).index_copy(0, referenced_indices, encoded_visual)
        else:
            unique_visual = self._encode_images(
                unique_images,
                unique_actor_boxes,
            )
        visual = unique_visual[safe_indices]
        return self._classify(visual, boxes, effective_mask)

    def backbone_frame_count(
        self,
        history_indices: Tensor | None,
        batch_size: int,
    ) -> int:
        if self.temporal_model == "v33-g":
            return 0
        if self.temporal_model != "v33-gr":
            if history_indices is None:
                return batch_size * len(self.history_offsets)
            return int(
                torch.unique(history_indices[history_indices.ge(0)]).numel()
            )
        if history_indices is None:
            return batch_size * len(self.visual_history_slot_indices)
        visual_slots = history_indices.new_tensor(
            self.visual_history_slot_indices
        )
        selected = history_indices.index_select(1, visual_slots)
        return int(torch.unique(selected[selected.ge(0)]).numel())
