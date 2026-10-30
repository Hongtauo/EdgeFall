"""V3.4 frozen-YOLO global/ROI visual confirmation model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from edgefall_st.models.bbox_classifier import (
    BBoxFallClassifier,
    CausalTemporalBlock,
    V32_LONG_HISTORY_OFFSETS,
    V32_SHORT_HISTORY_OFFSETS,
)


V34_VISUAL_SHORT_OFFSETS = (0, 2, 4, 7)
V34_VISUAL_LONG_OFFSETS = (0, 8, 16, 24, 32)
FYGR_FUSION_CONCAT_MLP = "concat-mlp"
FYGR_FUSION_CROSS_ATTENTION = "cross-attention-summary"
FYGR_FUSION_TEMPORAL_TOKENS = "temporal-visual-tokens"
FYGR_FUSION_MODES = (
    FYGR_FUSION_CONCAT_MLP,
    FYGR_FUSION_CROSS_ATTENTION,
    FYGR_FUSION_TEMPORAL_TOKENS,
)


def _checkpoint_arguments(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    arguments = checkpoint.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Geometry checkpoint is missing training arguments")
    return dict(arguments)


def load_v33_geometry(
    checkpoint_path: str | Path,
    *,
    history_offsets: Sequence[int],
    freeze: bool = True,
) -> BBoxFallClassifier:
    """Rebuild the exact V3.3-G branch stored in a checkpoint."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Geometry checkpoint does not contain model state")
    arguments = _checkpoint_arguments(checkpoint)
    short_offsets = tuple(
        int(value)
        for value in arguments.get(
            "classifier_v32_short_offsets", V32_SHORT_HISTORY_OFFSETS
        )
    )
    long_offsets = tuple(
        int(value)
        for value in arguments.get(
            "classifier_v32_long_offsets", V32_LONG_HISTORY_OFFSETS
        )
    )
    expected_history = tuple(sorted(set(short_offsets).union(long_offsets)))
    supplied_history = tuple(int(value) for value in history_offsets)
    if supplied_history != expected_history:
        raise ValueError(
            "Feature cache history offsets do not match the V3.3-G checkpoint: "
            f"{supplied_history} != {expected_history}"
        )
    model = BBoxFallClassifier(
        input_size=(
            int(arguments.get("input_height", 288)),
            int(arguments.get("input_width", 512)),
        ),
        crop_size=(
            int(arguments.get("crop_height", 128)),
            int(arguments.get("crop_width", 128)),
        ),
        crop_context=float(arguments.get("crop_context", 1.25)),
        visual_dim=int(arguments.get("classifier_visual_dim", 128)),
        temporal_dim=int(arguments.get("classifier_temporal_dim", 128)),
        history_offsets=supplied_history,
        temporal_model="v33-g",
        temporal_dropout=float(
            arguments.get("classifier_temporal_dropout", 0.1)
        ),
        short_history_offsets=short_offsets,
        long_history_offsets=long_offsets,
        effective_fps=float(arguments.get("effective_fps", 10.0)),
    )
    model.load_state_dict(state_dict, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze)
    model.train(not freeze)
    return model


def load_frozen_v33_geometry(
    checkpoint_path: str | Path,
    *,
    history_offsets: Sequence[int],
) -> BBoxFallClassifier:
    """Rebuild and freeze a V3.3-G branch for inference or mining."""

    return load_v33_geometry(
        checkpoint_path, history_offsets=history_offsets, freeze=True
    )


class _VisualTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        temporal_dim: int,
        visual_offsets: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or temporal_dim <= 0:
            raise ValueError("Visual dimensions must be positive")
        offsets = tuple(int(value) for value in visual_offsets)
        self.visual_offsets = offsets
        index = {offset: slot for slot, offset in enumerate(offsets)}
        for branch in (V34_VISUAL_SHORT_OFFSETS, V34_VISUAL_LONG_OFFSETS):
            if not set(branch).issubset(index):
                raise ValueError(
                    "visual_offsets must contain the V3.4 short/long anchors"
                )
        self.short_indices = tuple(index[value] for value in V34_VISUAL_SHORT_OFFSETS)
        self.long_indices = tuple(index[value] for value in V34_VISUAL_LONG_OFFSETS)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.SiLU(inplace=True),
        )
        self.short_positions = nn.Parameter(
            torch.zeros(len(self.short_indices), temporal_dim)
        )
        self.long_positions = nn.Parameter(
            torch.zeros(len(self.long_indices), temporal_dim)
        )
        nn.init.normal_(self.short_positions, std=0.02)
        nn.init.normal_(self.long_positions, std=0.02)
        self.short_temporal = nn.ModuleList(
            CausalTemporalBlock(
                temporal_dim, dilation=dilation, dropout=dropout
            )
            for dilation in (1, 2, 4)
        )
        self.long_temporal = nn.ModuleList(
            CausalTemporalBlock(
                temporal_dim, dilation=dilation, dropout=dropout
            )
            for dilation in (1, 2, 4)
        )
        self.summary = nn.Sequential(
            nn.Linear(4 * temporal_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _select(values: Tensor, indices: tuple[int, ...]) -> Tensor:
        return values.index_select(
            1, values.new_tensor(indices, dtype=torch.long)
        )

    def forward_with_tokens(
        self, features: Tensor, valid_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3 or valid_mask.shape != features.shape[:2]:
            raise ValueError("Expected features [B,V,C] and mask [B,V]")
        tokens = self.projection(features)
        # Cache slots are current-to-past. Causal convolutions consume the
        # reversed, chronological sequence so the last token is the present.
        short_mask = self._select(valid_mask, self.short_indices).flip(1)
        long_mask = self._select(valid_mask, self.long_indices).flip(1)
        short = self._select(tokens, self.short_indices).flip(1)
        long = self._select(tokens, self.long_indices).flip(1)
        short = (short + self.short_positions[None].to(short.dtype)) * short_mask[..., None]
        long = (long + self.long_positions[None].to(long.dtype)) * long_mask[..., None]
        short_current = short[:, -1]
        for block in self.short_temporal:
            short = block(short, short_mask)
        for block in self.long_temporal:
            long = block(long, long_mask)
        summary = self.summary(
            torch.cat(
                (
                    short_current,
                    short[:, -1],
                    long[:, -1],
                    short[:, -1] - long[:, -1],
                ),
                dim=-1,
            )
        )
        timeline, timeline_mask = BBoxFallClassifier._merge_temporal_branches(
            (
                (short, short_mask, V34_VISUAL_SHORT_OFFSETS),
                (long, long_mask, V34_VISUAL_LONG_OFFSETS),
            ),
            self.visual_offsets,
        )
        return summary, timeline, timeline_mask

    def forward(self, features: Tensor, valid_mask: Tensor) -> Tensor:
        summary, _, _ = self.forward_with_tokens(features, valid_mask)
        return summary


class _GeometryConditionedVisualAttention(nn.Module):
    """Retrieve mandatory visual evidence using geometry as the query."""

    def __init__(
        self,
        geometry_dim: int,
        *,
        temporal_dim: int,
        modality_count: int,
        attention_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(geometry_dim, temporal_dim, modality_count) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if attention_heads <= 0 or temporal_dim % attention_heads:
            raise ValueError(
                "temporal_dim must be divisible by attention_heads"
            )
        if ffn_dim <= 0:
            raise ValueError("attention_ffn_dim must be positive")
        self.query_projection = nn.Sequential(
            nn.Linear(geometry_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
        )
        self.modality_positions = nn.Parameter(
            torch.zeros(modality_count, temporal_dim)
        )
        nn.init.normal_(self.modality_positions, std=0.02)
        self.attention = nn.MultiheadAttention(
            temporal_dim,
            attention_heads,
            # Dropping the only valid modality would create an implicit null
            # visual path, so regularization stays in the visual FFN instead.
            dropout=0.0,
            batch_first=True,
        )
        self.visual_ffn = nn.Sequential(
            nn.Linear(temporal_dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, temporal_dim),
            nn.Dropout(dropout),
        )
        self.visual_norm = nn.LayerNorm(temporal_dim)
        self.interaction = nn.Sequential(
            nn.Linear(3 * temporal_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        geometry_state: Tensor,
        visual_states: Sequence[Tensor],
        visual_valid: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if len(visual_states) != self.modality_positions.shape[0]:
            raise ValueError("Visual state count does not match modalities")
        if len(visual_states) != len(visual_valid):
            raise ValueError("Visual states and validity masks disagree")
        tokens = torch.stack(tuple(visual_states), dim=1)
        token_valid = torch.stack(tuple(visual_valid), dim=1).bool()
        if not bool(token_valid.any(dim=1).all()):
            raise ValueError(
                "Cross-attention requires at least one valid visual token "
                "per sample"
            )
        query = self.query_projection(geometry_state)
        keys = tokens + self.modality_positions[None].to(tokens.dtype)
        evidence, attention = self.attention(
            query[:, None],
            keys,
            tokens,
            key_padding_mask=~token_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        evidence = evidence[:, 0]
        visual_state = self.visual_norm(
            evidence + self.visual_ffn(evidence)
        )
        fused = self.interaction(
            torch.cat(
                (
                    visual_state,
                    query * visual_state,
                    torch.abs(query - visual_state),
                ),
                dim=-1,
            )
        )
        # [B, heads, query=1, modalities] -> [B, modalities]
        weights = attention[:, :, 0].mean(dim=1)
        return fused, weights


class _GeometryTemporalVisualAttention(nn.Module):
    """Use a geometry timeline to retrieve causal, mandatory visual evidence."""

    def __init__(
        self,
        geometry_dim: int,
        *,
        temporal_dim: int,
        history_offsets: Sequence[int],
        visual_offsets: Sequence[int],
        modality_count: int,
        attention_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(geometry_dim, temporal_dim, modality_count) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if attention_heads <= 0 or temporal_dim % attention_heads:
            raise ValueError(
                "temporal_dim must be divisible by attention_heads"
            )
        if ffn_dim <= 0:
            raise ValueError("attention_ffn_dim must be positive")
        self.attention_heads = int(attention_heads)
        self.query_offsets = tuple(
            reversed(tuple(int(value) for value in history_offsets))
        )
        self.visual_offsets = tuple(
            reversed(tuple(int(value) for value in visual_offsets))
        )
        self.query_projection = nn.Sequential(
            nn.Linear(geometry_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
        )
        self.query_positions = nn.Parameter(
            torch.zeros(len(self.query_offsets), temporal_dim)
        )
        self.visual_positions = nn.Parameter(
            torch.zeros(len(self.visual_offsets), temporal_dim)
        )
        self.modality_positions = nn.Parameter(
            torch.zeros(modality_count, temporal_dim)
        )
        nn.init.normal_(self.query_positions, std=0.02)
        nn.init.normal_(self.visual_positions, std=0.02)
        nn.init.normal_(self.modality_positions, std=0.02)
        self.attention = nn.MultiheadAttention(
            temporal_dim,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.visual_ffn = nn.Sequential(
            nn.Linear(temporal_dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, temporal_dim),
            nn.Dropout(dropout),
        )
        self.visual_norm = nn.LayerNorm(temporal_dim)
        self.temporal = nn.ModuleList(
            CausalTemporalBlock(
                temporal_dim, dilation=dilation, dropout=dropout
            )
            for dilation in (1, 2)
        )
        self.output_norm = nn.LayerNorm(temporal_dim)
        causal = torch.tensor(
            [
                [visual_offset >= query_offset for visual_offset in self.visual_offsets]
                * modality_count
                for query_offset in self.query_offsets
            ],
            dtype=torch.bool,
        )
        self.register_buffer("causal_visual_mask", causal, persistent=False)

    @staticmethod
    def _carry_forward(
        tokens: Tensor, valid_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        state = torch.zeros_like(tokens[:, 0])
        has_state = torch.zeros(
            tokens.shape[0], dtype=torch.bool, device=tokens.device
        )
        carried: list[Tensor] = []
        carried_valid: list[Tensor] = []
        for index in range(tokens.shape[1]):
            current_valid = valid_mask[:, index].bool()
            state = torch.where(current_valid[:, None], tokens[:, index], state)
            has_state = has_state | current_valid
            carried.append(state)
            carried_valid.append(has_state)
        return torch.stack(carried, dim=1), torch.stack(
            carried_valid, dim=1
        )

    def forward(
        self,
        geometry_tokens: Tensor,
        geometry_valid: Tensor,
        visual_tokens: Sequence[Tensor],
        visual_valid: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if geometry_tokens.ndim != 3:
            raise ValueError("geometry_tokens must have shape [B,Q,C]")
        if geometry_valid.shape != geometry_tokens.shape[:2]:
            raise ValueError("Geometry tokens and validity mask disagree")
        if geometry_tokens.shape[1] != len(self.query_offsets):
            raise ValueError("Unexpected geometry query count")
        if len(visual_tokens) != self.modality_positions.shape[0]:
            raise ValueError("Visual token count does not match modalities")
        if len(visual_tokens) != len(visual_valid):
            raise ValueError("Visual tokens and validity masks disagree")

        carried_geometry, query_valid = self._carry_forward(
            geometry_tokens, geometry_valid
        )
        no_geometry = ~query_valid.any(dim=1)
        if bool(no_geometry.any()):
            query_valid = query_valid.clone()
            query_valid[no_geometry, -1] = True
        query = self.query_projection(carried_geometry)
        query = query + self.query_positions[None].to(query.dtype)
        keys: list[Tensor] = []
        values: list[Tensor] = []
        key_masks: list[Tensor] = []
        for modality, (tokens, valid) in enumerate(
            zip(visual_tokens, visual_valid, strict=True)
        ):
            if tokens.ndim != 3 or tokens.shape[1] != len(self.visual_offsets):
                raise ValueError("Unexpected visual token timeline")
            if valid.shape != tokens.shape[:2]:
                raise ValueError("Visual tokens and validity mask disagree")
            keys.append(
                tokens
                + self.visual_positions[None].to(tokens.dtype)
                + self.modality_positions[modality][None, None].to(tokens.dtype)
            )
            values.append(tokens)
            key_masks.append(valid.bool())
        key = torch.cat(keys, dim=1)
        value = torch.cat(values, dim=1)
        key_valid = torch.cat(key_masks, dim=1)

        allowed = (
            self.causal_visual_mask[None]
            & key_valid[:, None]
            & query_valid[:, :, None]
        )
        row_valid = allowed.any(dim=-1)
        if not bool(row_valid[:, -1].all()):
            raise ValueError(
                "Temporal fusion requires visual evidence for the current query"
            )
        safe_allowed = allowed.clone()
        safe_allowed[:, :, 0] |= ~row_valid
        attention_mask = (
            ~safe_allowed[:, None]
            .expand(-1, self.attention_heads, -1, -1)
            .reshape(
                geometry_tokens.shape[0] * self.attention_heads,
                geometry_tokens.shape[1],
                key.shape[1],
            )
        )
        evidence, attention = self.attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        evidence = evidence * row_valid[:, :, None].to(evidence.dtype)
        visual_state = self.visual_norm(
            evidence + self.visual_ffn(evidence)
        ) * row_valid[:, :, None].to(evidence.dtype)
        for block in self.temporal:
            visual_state = block(visual_state, row_valid)
        pre_norm_fused = visual_state[:, -1]
        fused = self.output_norm(pre_norm_fused)
        weights = attention.mean(dim=1)
        weights = weights * row_valid[:, :, None].to(weights.dtype)
        return fused, weights, row_valid, key_valid, pre_norm_fused


class FrozenYoloGlobalROIConfirmer(nn.Module):
    """Directly reclassify a frozen V3.3-G state using frozen YOLO cues.

    This is deliberately not a residual gate. The confirmation heads receive
    geometry, ROI, and scene states and produce a new event decision with
    independent visual supervision.
    """

    def __init__(
        self,
        *,
        geometry_checkpoint: str | Path,
        history_offsets: Sequence[int],
        visual_offsets: Sequence[int],
        global_feature_dim: int,
        roi_feature_shape: Sequence[int],
        temporal_dim: int = 128,
        dropout: float = 0.1,
        use_global: bool = True,
        use_roi: bool = True,
        freeze_geometry: bool = True,
        fusion_mode: str = FYGR_FUSION_CONCAT_MLP,
        attention_heads: int = 4,
        attention_ffn_dim: int = 256,
        person_mil_temperature: float = 0.0,
        num_actions: int = 16,
    ) -> None:
        super().__init__()
        if not use_global and not use_roi:
            raise ValueError("At least one visual modality must be enabled")
        roi_shape = tuple(int(value) for value in roi_feature_shape)
        if len(roi_shape) != 3 or any(value <= 0 for value in roi_shape):
            raise ValueError("roi_feature_shape must be [C,H,W]")
        self.history_offsets = tuple(int(value) for value in history_offsets)
        self.visual_offsets = tuple(int(value) for value in visual_offsets)
        self.use_global = bool(use_global)
        self.use_roi = bool(use_roi)
        self.freeze_geometry = bool(freeze_geometry)
        self.fusion_mode = str(fusion_mode)
        self.person_mil_temperature = float(person_mil_temperature)
        if self.person_mil_temperature < 0:
            raise ValueError("person_mil_temperature cannot be negative")
        if self.fusion_mode not in FYGR_FUSION_MODES:
            raise ValueError(
                f"Unsupported fusion_mode {self.fusion_mode!r}; "
                f"expected one of {FYGR_FUSION_MODES}"
            )
        self.geometry = load_v33_geometry(
            geometry_checkpoint,
            history_offsets=self.history_offsets,
            freeze=self.freeze_geometry,
        )
        geometry_dim = int(self.geometry.stage_head.in_features)
        states: list[int] = [geometry_dim]
        if self.use_global:
            self.global_encoder = _VisualTemporalEncoder(
                int(global_feature_dim),
                temporal_dim=temporal_dim,
                visual_offsets=self.visual_offsets,
                dropout=dropout,
            )
            self.global_event_head = nn.Linear(temporal_dim, 2)
            states.append(temporal_dim)
        else:
            self.global_encoder = None
            self.global_event_head = None
        if self.use_roi:
            self.roi_encoder = _VisualTemporalEncoder(
                int(torch.tensor(roi_shape).prod().item()),
                temporal_dim=temporal_dim,
                visual_offsets=self.visual_offsets,
                dropout=dropout,
            )
            self.roi_event_head = nn.Linear(temporal_dim, 2)
            states.append(temporal_dim)
        else:
            self.roi_encoder = None
            self.roi_event_head = None
        self.temporal_cross_attention: (
            _GeometryTemporalVisualAttention | None
        ) = None
        if self.fusion_mode == FYGR_FUSION_CONCAT_MLP:
            fusion_dim = sum(states)
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.cross_attention = None
        elif self.fusion_mode == FYGR_FUSION_CROSS_ATTENTION:
            self.fusion = None
            self.cross_attention = _GeometryConditionedVisualAttention(
                geometry_dim,
                temporal_dim=temporal_dim,
                modality_count=int(self.use_global) + int(self.use_roi),
                attention_heads=attention_heads,
                ffn_dim=attention_ffn_dim,
                dropout=dropout,
            )
        else:
            self.fusion = None
            self.cross_attention = None
            self.temporal_cross_attention = _GeometryTemporalVisualAttention(
                geometry_dim,
                temporal_dim=temporal_dim,
                history_offsets=self.history_offsets,
                visual_offsets=self.visual_offsets,
                modality_count=int(self.use_global) + int(self.use_roi),
                attention_heads=attention_heads,
                ffn_dim=attention_ffn_dim,
                dropout=dropout,
            )
        self.event_state_head = nn.Linear(temporal_dim, 2)
        self.stage_head = nn.Linear(temporal_dim, 3)
        self.onset_head = nn.Linear(temporal_dim, 1)
        self.action_head = nn.Linear(temporal_dim, num_actions)
        self.control_head = nn.Linear(temporal_dim, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> FrozenYoloGlobalROIConfirmer:
        super().train(mode)
        if self.freeze_geometry:
            self.geometry.eval()
        return self

    @staticmethod
    def _event_logit(state_logits: Tensor) -> Tensor:
        return BBoxFallClassifier._event_logit(state_logits)

    def _forward_single_person(
        self,
        global_features: Tensor,
        roi_features: Tensor,
        actor_boxes: Tensor,
        valid_mask: Tensor,
        actor_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if global_features.ndim != 3:
            raise ValueError("global_features must have shape [B,V,C]")
        if roi_features.ndim != 5:
            raise ValueError("roi_features must have shape [B,V,C,H,W]")
        if global_features.shape[:2] != roi_features.shape[:2]:
            raise ValueError("Global and ROI feature timelines disagree")
        visual_slots = global_features.shape[1]
        if visual_slots != len(self.visual_offsets):
            raise ValueError("Unexpected number of visual feature slots")
        if self.freeze_geometry:
            with torch.no_grad():
                geometry = self.geometry.forward_geometry_history(
                    actor_boxes, valid_mask, actor_valid_mask
                )
            geometry_state = geometry["geometry_state"].detach()
        else:
            geometry = self.geometry.forward_geometry_history(
                actor_boxes, valid_mask, actor_valid_mask
            )
            geometry_state = geometry["geometry_state"]
        states = [geometry_state]
        visual_states: list[Tensor] = []
        visual_state_valid: list[Tensor] = []
        visual_tokens: list[Tensor] = []
        visual_token_valid: list[Tensor] = []
        outputs: dict[str, Tensor] = {}
        visual_indices = actor_boxes.new_tensor(
            [self.history_offsets.index(value) for value in self.visual_offsets],
            dtype=torch.long,
        )
        global_mask = valid_mask.index_select(1, visual_indices)
        roi_mask = global_mask * actor_valid_mask.index_select(1, visual_indices)
        if self.global_encoder is not None and self.global_event_head is not None:
            global_state, global_tokens, global_token_mask = (
                self.global_encoder.forward_with_tokens(
                    global_features, global_mask
                )
            )
            states.append(global_state)
            visual_states.append(global_state)
            visual_state_valid.append(global_mask.bool().any(dim=1))
            visual_tokens.append(global_tokens)
            visual_token_valid.append(global_token_mask)
            global_event_logits = self.global_event_head(global_state)
            outputs["global_fall_logits"] = self._event_logit(
                global_event_logits
            )
        if self.roi_encoder is not None and self.roi_event_head is not None:
            roi_state, roi_tokens, roi_token_mask = (
                self.roi_encoder.forward_with_tokens(
                    roi_features.flatten(2), roi_mask
                )
            )
            states.append(roi_state)
            visual_states.append(roi_state)
            visual_state_valid.append(roi_mask.bool().any(dim=1))
            visual_tokens.append(roi_tokens)
            visual_token_valid.append(roi_token_mask)
            roi_event_logits = self.roi_event_head(roi_state)
            outputs["roi_fall_logits"] = self._event_logit(roi_event_logits)
        if self.fusion is not None:
            fused = self.fusion(torch.cat(states, dim=-1))
        elif self.cross_attention is not None:
            fused, attention_weights = self.cross_attention(
                geometry_state,
                visual_states,
                visual_state_valid,
            )
            outputs["fusion_attention_weights"] = attention_weights
            outputs["fusion_key_valid_mask"] = torch.stack(
                tuple(visual_state_valid), dim=1
            )
        else:
            if self.temporal_cross_attention is None:
                raise RuntimeError("Temporal cross-attention is not initialized")
            (
                fused,
                attention_weights,
                query_valid,
                key_valid,
                pre_norm_fused,
            ) = self.temporal_cross_attention(
                geometry["geometry_tokens"],
                geometry["geometry_token_mask"],
                visual_tokens,
                visual_token_valid,
            )
            outputs["fusion_attention_weights"] = attention_weights
            outputs["fusion_query_valid_mask"] = query_valid
            outputs["fusion_key_valid_mask"] = key_valid
            outputs["fusion_pre_norm_state"] = pre_norm_fused
        outputs["fusion_state"] = fused
        event_state_logits = self.event_state_head(fused)
        outputs.update(
            {
                "stage_logits": self.stage_head(fused),
                "fall_logits": self._event_logit(event_state_logits),
                "falling_logits": event_state_logits[:, 0],
                "fallen_logits": event_state_logits[:, 1],
                "onset_logits": self.onset_head(fused).squeeze(-1),
                "action_logits": self.action_head(fused),
                "control_logits": self.control_head(fused).squeeze(-1),
                "geometry_stage_logits": geometry["geometry_stage_logits"],
                "geometry_fall_logits": geometry["geometry_fall_logits"],
                "geometry_falling_logits": geometry[
                    "geometry_falling_logits"
                ],
                "geometry_fallen_logits": geometry["geometry_fallen_logits"],
            }
        )
        return outputs

    @staticmethod
    def _gather_person(values: Tensor, indices: Tensor) -> Tensor:
        view = indices.view(indices.shape[0], 1, *([1] * (values.ndim - 2)))
        expanded = view.expand(indices.shape[0], 1, *values.shape[2:])
        return values.gather(1, expanded).squeeze(1)

    def _aggregate_people(self, values: Tensor, valid: Tensor) -> Tensor:
        minimum = torch.finfo(values.dtype).min
        masked = values.masked_fill(~valid, minimum)
        if self.person_mil_temperature == 0.0:
            return masked.max(dim=1).values
        temperature = self.person_mil_temperature
        count = valid.sum(dim=1).clamp_min(1).to(values.dtype)
        return temperature * (
            torch.logsumexp(masked / temperature, dim=1)
            - count.log()
        )

    def forward(
        self,
        global_features: Tensor,
        roi_features: Tensor,
        actor_boxes: Tensor,
        valid_mask: Tensor,
        actor_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Score every cached person and aggregate the bag with max-MIL."""

        if global_features.ndim != 3:
            raise ValueError("global_features must have shape [B,V,C]")
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape [B,H]")
        if actor_boxes.ndim == 3:
            actor_boxes = actor_boxes.unsqueeze(2)
            actor_valid_mask = actor_valid_mask.unsqueeze(2)
            roi_features = roi_features.unsqueeze(2)
        if actor_boxes.ndim != 4 or actor_boxes.shape[-1] != 4:
            raise ValueError("actor_boxes must have shape [B,H,P,4]")
        if actor_valid_mask.shape != actor_boxes.shape[:-1]:
            raise ValueError("actor_valid_mask shape disagrees with actor_boxes")
        if roi_features.ndim != 6:
            raise ValueError("roi_features must have shape [B,V,P,C,H,W]")
        batch_size, history_slots, person_count, _ = actor_boxes.shape
        if roi_features.shape[:3] != (
            batch_size,
            global_features.shape[1],
            person_count,
        ):
            raise ValueError("ROI, geometry, and global timelines disagree")
        if valid_mask.shape != (batch_size, history_slots):
            raise ValueError("valid_mask shape disagrees with actor_boxes")

        person_valid = actor_valid_mask.gt(0).any(dim=1)
        no_valid_person = ~person_valid.any(dim=1)
        if bool(no_valid_person.any()):
            person_valid = person_valid.clone()
            person_valid[no_valid_person, 0] = True
        valid_person_indices = torch.nonzero(
            person_valid.reshape(-1), as_tuple=False
        ).flatten()

        visual_slots = int(global_features.shape[1])
        flat_global = (
            global_features[:, None]
            .expand(batch_size, person_count, visual_slots, global_features.shape[-1])
            .reshape(batch_size * person_count, visual_slots, global_features.shape[-1])
        )
        flat_roi = roi_features.permute(0, 2, 1, 3, 4, 5).reshape(
            batch_size * person_count,
            visual_slots,
            *roi_features.shape[3:],
        )
        flat_boxes = actor_boxes.permute(0, 2, 1, 3).reshape(
            batch_size * person_count, history_slots, 4
        )
        flat_valid = (
            valid_mask[:, None]
            .expand(batch_size, person_count, history_slots)
            .reshape(batch_size * person_count, history_slots)
        )
        flat_actor_valid = actor_valid_mask.permute(0, 2, 1).reshape(
            batch_size * person_count, history_slots
        )
        valid_outputs = self._forward_single_person(
            flat_global.index_select(0, valid_person_indices),
            flat_roi.index_select(0, valid_person_indices),
            flat_boxes.index_select(0, valid_person_indices),
            flat_valid.index_select(0, valid_person_indices),
            flat_actor_valid.index_select(0, valid_person_indices),
        )

        person_outputs: dict[str, Tensor] = {}
        for name, value in valid_outputs.items():
            flat_value = value.new_zeros(
                batch_size * person_count, *value.shape[1:]
            ).index_copy(0, valid_person_indices, value)
            person_outputs[name] = flat_value.reshape(
                batch_size, person_count, *value.shape[1:]
            )
        person_fall_logits = person_outputs["fall_logits"]
        masked_fall_logits = person_fall_logits.masked_fill(
            ~person_valid, torch.finfo(person_fall_logits.dtype).min
        )
        selected_person = masked_fall_logits.argmax(dim=1)

        outputs: dict[str, Tensor] = {}
        for name, values in person_outputs.items():
            if name == "action_logits":
                # The 16-class action target belongs to the original primary
                # actor path, not to an unlabeled alternate person.
                outputs[name] = values[:, 0]
            elif values.ndim == 2:
                outputs[name] = self._aggregate_people(
                    values, person_valid
                )
            else:
                outputs[name] = self._gather_person(values, selected_person)
            outputs[f"person_{name}"] = values
        outputs["person_valid_mask"] = person_valid
        outputs["selected_person_index"] = selected_person
        return outputs
