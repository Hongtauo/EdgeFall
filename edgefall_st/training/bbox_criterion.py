"""Losses for bbox-conditioned fall-event classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONTROLLED_DESCENT_ACTION_IDS = (3, 5, 10, 12)


@dataclass(frozen=True)
class BBoxClassifierLossWeights:
    stage: float = 1.0
    fall: float = 2.0
    event_state: float = 0.0
    onset: float = 0.5
    action: float = 0.0
    control: float = 0.0
    geometry_stage: float = 0.0
    geometry_fall: float = 0.0
    consistency: float = 0.25
    video_mil: float = 0.5
    video_ranking: float = 0.25
    hard_video_ranking: float = 0.0


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


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


class BBoxFallCriterion(nn.Module):
    """Optimize semantic fall discrimination without localization losses."""

    def __init__(
        self,
        *,
        loss_weights: BBoxClassifierLossWeights = (
            BBoxClassifierLossWeights()
        ),
        stage_class_weights: Sequence[float] = (1.0, 2.0, 2.0),
        action_class_weights: Sequence[float] = (1.0,) * 16,
        stage_gamma: float = 2.0,
        fall_alpha: float = 0.75,
        fall_gamma: float = 2.0,
        onset_alpha: float = 0.90,
        onset_gamma: float = 2.0,
        video_ranking_margin: float = 0.2,
        hard_video_ranking_margin: float = 0.2,
        hard_negative_window_keys: Sequence[tuple[str, int]] = (),
        video_pooling: str = "max",
        video_topk: int = 2,
        control_alpha: float = 0.75,
        control_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if len(stage_class_weights) != 3:
            raise ValueError("stage_class_weights must have three values")
        if len(action_class_weights) != 16:
            raise ValueError("action_class_weights must have sixteen values")
        if not 0.0 <= fall_alpha <= 1.0:
            raise ValueError("fall_alpha must be in [0, 1]")
        if not 0.0 <= onset_alpha <= 1.0:
            raise ValueError("onset_alpha must be in [0, 1]")
        if not 0.0 <= control_alpha <= 1.0:
            raise ValueError("control_alpha must be in [0, 1]")
        if min(stage_gamma, fall_gamma, onset_gamma, control_gamma) < 0:
            raise ValueError("Focal gamma values cannot be negative")
        if video_ranking_margin < 0:
            raise ValueError("video_ranking_margin cannot be negative")
        if hard_video_ranking_margin < 0:
            raise ValueError("hard_video_ranking_margin cannot be negative")
        if video_pooling not in {"max", "topk-mean", "asymmetric"}:
            raise ValueError(
                "video_pooling must be 'max', 'topk-mean', or 'asymmetric'"
            )
        if video_topk <= 0:
            raise ValueError("video_topk must be positive")
        if any(
            value < 0
            for value in (
                loss_weights.stage,
                loss_weights.fall,
                loss_weights.event_state,
                loss_weights.onset,
                loss_weights.action,
                loss_weights.control,
                loss_weights.geometry_stage,
                loss_weights.geometry_fall,
                loss_weights.consistency,
                loss_weights.video_mil,
                loss_weights.video_ranking,
                loss_weights.hard_video_ranking,
            )
        ):
            raise ValueError("Loss weights cannot be negative")

        self.loss_weights = loss_weights
        self.stage_gamma = float(stage_gamma)
        self.fall_alpha = float(fall_alpha)
        self.fall_gamma = float(fall_gamma)
        self.onset_alpha = float(onset_alpha)
        self.onset_gamma = float(onset_gamma)
        self.control_alpha = float(control_alpha)
        self.control_gamma = float(control_gamma)
        self.video_ranking_margin = float(video_ranking_margin)
        self.hard_video_ranking_margin = float(
            hard_video_ranking_margin
        )
        self.hard_negative_window_keys = {
            (str(path), int(window_index))
            for path, window_index in hard_negative_window_keys
        }
        self.video_pooling = video_pooling
        self.video_topk = int(video_topk)
        self.register_buffer(
            "stage_class_weights",
            torch.tensor(stage_class_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "action_class_weights",
            torch.tensor(action_class_weights, dtype=torch.float32),
        )

    def _pool_video_logits(
        self,
        logits: Tensor,
        *,
        positive: bool = False,
        weak_positive: bool = False,
    ) -> Tensor:
        if self.video_pooling == "max":
            return logits.max()
        if self.video_pooling == "asymmetric":
            if positive and weak_positive:
                count = min(self.video_topk, int(logits.numel()))
                return logits.topk(count, sorted=False).values.mean()
            return logits.mean() if positive else logits.max()
        count = min(self.video_topk, int(logits.numel()))
        return logits.topk(count, sorted=False).values.mean()

    def _video_objectives(
        self,
        fall_logits: Tensor,
        metadata: object | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        zero = fall_logits.sum() * 0.0
        zero_count = fall_logits.new_zeros((), dtype=torch.long)
        if (
            not isinstance(metadata, Sequence)
            or isinstance(metadata, (str, bytes))
            or len(metadata) != fall_logits.shape[0]
        ):
            return zero, zero, zero, zero_count, zero_count, zero_count

        grouped_indices: dict[str, list[int]] = {}
        grouped_metadata: dict[str, list[Mapping[str, object]]] = {}
        for index, item in enumerate(metadata):
            if not isinstance(item, Mapping) or "path" not in item:
                return zero, zero, zero, zero_count, zero_count, zero_count
            path = str(item["path"])
            grouped_indices.setdefault(path, []).append(index)
            grouped_metadata.setdefault(path, []).append(item)

        group_scores: list[Tensor] = []
        group_targets: list[float] = []
        group_hard_negative: list[bool] = []
        for path, indices in grouped_indices.items():
            items = grouped_metadata[path]
            if not all("video_has_fall_event" in item for item in items):
                continue
            video_has_fall = bool(items[0]["video_has_fall_event"])
            if any(
                bool(item["video_has_fall_event"]) != video_has_fall
                for item in items[1:]
            ):
                raise ValueError(
                    f"Inconsistent video_has_fall_event for {path!r}"
                )
            weak_video_label = bool(items[0].get("weak_video_label", False))
            if any(
                bool(item.get("weak_video_label", False)) != weak_video_label
                for item in items[1:]
            ):
                raise ValueError(f"Inconsistent weak label mode for {path!r}")
            chunk_has_event = any(
                bool(
                    item.get(
                        "event_target",
                        int(item.get("window_action_label", -1)) in (1, 2),
                    )
                )
                for item in items
            )
            if video_has_fall and not weak_video_label and not chunk_has_event:
                # A normal-only chunk from a positive video is irrelevant to
                # video-level max pooling, but remains window-supervised.
                continue
            pooled_indices = indices
            if (
                video_has_fall
                and not weak_video_label
                and self.video_pooling == "asymmetric"
            ):
                event_indices = [
                    index
                    for index, item in zip(indices, items, strict=True)
                    if bool(
                        item.get(
                            "event_target",
                            int(item.get("window_action_label", -1))
                            in (1, 2),
                        )
                    )
                ]
                if event_indices:
                    pooled_indices = event_indices
            index_tensor = torch.tensor(
                pooled_indices,
                device=fall_logits.device,
            )
            group_scores.append(
                self._pool_video_logits(
                    fall_logits[index_tensor],
                    positive=video_has_fall,
                    weak_positive=video_has_fall and weak_video_label,
                )
            )
            group_targets.append(float(video_has_fall))
            group_hard_negative.append(
                not video_has_fall
                and any(
                    (
                        path,
                        int(item.get("window_index", -1)),
                    )
                    in self.hard_negative_window_keys
                    for item in items
                )
            )

        if not group_scores:
            return zero, zero, zero, zero_count, zero_count, zero_count
        scores = torch.stack(group_scores)
        targets = scores.new_tensor(group_targets)
        hard_negative_mask = torch.tensor(
            group_hard_negative,
            device=scores.device,
            dtype=torch.bool,
        )
        loss_video_mil = _sigmoid_focal_loss(
            scores,
            targets,
            alpha=0.5,
            gamma=2.0,
        ).mean()

        positive_scores = scores[targets > 0.5]
        negative_scores = scores[targets <= 0.5]
        if positive_scores.numel() and negative_scores.numel():
            loss_video_ranking = F.relu(
                self.video_ranking_margin
                - positive_scores[:, None]
                + negative_scores[None, :]
            ).mean()
        else:
            loss_video_ranking = zero
        hard_negative_scores = scores[hard_negative_mask]
        if positive_scores.numel() and hard_negative_scores.numel():
            loss_hard_video_ranking = F.relu(
                self.hard_video_ranking_margin
                - positive_scores[:, None]
                + hard_negative_scores[None, :]
            ).mean()
        else:
            loss_hard_video_ranking = zero
        return (
            loss_video_mil,
            loss_video_ranking,
            loss_hard_video_ranking,
            scores.new_tensor(scores.numel(), dtype=torch.long),
            scores.new_tensor(positive_scores.numel(), dtype=torch.long),
            scores.new_tensor(
                hard_negative_scores.numel(), dtype=torch.long
            ),
        )

    def _validate_outputs(
        self,
        outputs: dict[str, Tensor],
        batch_size: int,
    ) -> None:
        expected = {
            "stage_logits": (batch_size, 3),
            "fall_logits": (batch_size,),
            "onset_logits": (batch_size,),
            "action_logits": (batch_size, 16),
        }
        if self.loss_weights.control > 0:
            expected["control_logits"] = (batch_size,)
        if self.loss_weights.event_state > 0:
            expected["falling_logits"] = (batch_size,)
            expected["fallen_logits"] = (batch_size,)
        if self.loss_weights.geometry_stage > 0:
            expected["geometry_stage_logits"] = (batch_size, 3)
        if self.loss_weights.geometry_fall > 0:
            expected["geometry_fall_logits"] = (batch_size,)
        for name, shape in expected.items():
            if name not in outputs:
                raise ValueError(f"Model outputs are missing {name!r}")
            if outputs[name].shape != shape:
                raise ValueError(
                    f"outputs[{name!r}] must have shape {shape}"
                )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        metadata: object | None = None,
    ) -> dict[str, Tensor]:
        batch_size = len(targets)
        self._validate_outputs(outputs, batch_size)
        stage_logits = outputs["stage_logits"].float()
        fall_logits = outputs["fall_logits"].float()
        onset_logits = outputs["onset_logits"].float()
        action_logits = outputs["action_logits"].float()

        stage_targets = torch.stack(
            [target["stage_label"] for target in targets]
        ).long()
        action_targets = torch.stack(
            [target["action_label"] for target in targets]
        ).long()
        onset_targets = torch.stack(
            [target["onset_target"] for target in targets]
        ).float()
        event_weights = torch.stack(
            [
                target.get("event_loss_weight", target["action_loss_weight"])
                for target in targets
            ]
        ).float()
        action_weights = torch.stack(
            [target["action_loss_weight"] for target in targets]
        ).float()
        control_weights = torch.stack(
            [
                target.get("control_loss_weight", target["action_loss_weight"])
                for target in targets
            ]
        ).float()
        stage_fall_targets = stage_targets.ne(0)
        fall_targets = torch.stack(
            [
                target.get(
                    "event_target",
                    target["stage_label"].ne(0),
                )
                for target in targets
            ]
        ).to(dtype=fall_logits.dtype)

        stage_cross_entropy = F.cross_entropy(
            stage_logits,
            stage_targets,
            weight=self.stage_class_weights.to(stage_logits.dtype),
            reduction="none",
        )
        target_stage_probability = stage_logits.softmax(dim=-1).gather(
            1,
            stage_targets[:, None],
        ).squeeze(1)
        loss_stage = _weighted_mean(
            (1.0 - target_stage_probability).pow(self.stage_gamma)
            * stage_cross_entropy,
            event_weights,
        )
        loss_fall = _weighted_mean(
            _sigmoid_focal_loss(
                fall_logits,
                fall_targets,
                alpha=self.fall_alpha,
                gamma=self.fall_gamma,
            ),
            event_weights,
        )
        if self.loss_weights.event_state > 0:
            falling_targets = (
                fall_targets * stage_targets.eq(1).to(fall_targets.dtype)
            )
            fallen_targets = (
                fall_targets * stage_targets.eq(2).to(fall_targets.dtype)
            )
            loss_event_state = 0.5 * (
                _weighted_mean(
                    _sigmoid_focal_loss(
                        outputs["falling_logits"].float(),
                        falling_targets,
                        alpha=self.fall_alpha,
                        gamma=self.fall_gamma,
                    ),
                    event_weights,
                )
                + _weighted_mean(
                    _sigmoid_focal_loss(
                        outputs["fallen_logits"].float(),
                        fallen_targets,
                        alpha=self.fall_alpha,
                        gamma=self.fall_gamma,
                    ),
                    event_weights,
                )
            )
        else:
            loss_event_state = fall_logits.sum() * 0.0
        if self.loss_weights.geometry_stage > 0:
            geometry_stage_logits = outputs["geometry_stage_logits"].float()
            geometry_stage_cross_entropy = F.cross_entropy(
                geometry_stage_logits,
                stage_targets,
                weight=self.stage_class_weights.to(
                    geometry_stage_logits.dtype
                ),
                reduction="none",
            )
            geometry_target_probability = geometry_stage_logits.softmax(
                dim=-1
            ).gather(1, stage_targets[:, None]).squeeze(1)
            loss_geometry_stage = _weighted_mean(
                (1.0 - geometry_target_probability).pow(self.stage_gamma)
                * geometry_stage_cross_entropy,
                event_weights,
            )
        else:
            loss_geometry_stage = fall_logits.sum() * 0.0
        if self.loss_weights.geometry_fall > 0:
            loss_geometry_fall = _weighted_mean(
                _sigmoid_focal_loss(
                    outputs["geometry_fall_logits"].float(),
                    fall_targets,
                    alpha=self.fall_alpha,
                    gamma=self.fall_gamma,
                ),
                event_weights,
            )
        else:
            loss_geometry_fall = fall_logits.sum() * 0.0
        loss_onset = _weighted_mean(
            _sigmoid_focal_loss(
                onset_logits,
                onset_targets,
                alpha=self.onset_alpha,
                gamma=self.onset_gamma,
            ),
            event_weights,
        )
        stage_fall_probabilities = 1.0 - stage_logits.softmax(dim=-1)[:, 0]
        fall_probabilities = fall_logits.sigmoid()
        consistency_weights = event_weights * fall_targets.eq(
            stage_fall_targets
        ).to(event_weights.dtype)
        loss_consistency = _weighted_mean(
            (fall_probabilities - stage_fall_probabilities).square(),
            consistency_weights,
        )
        (
            loss_video_mil,
            loss_video_ranking,
            loss_hard_video_ranking,
            num_video_groups,
            num_video_positive_groups,
            num_hard_negative_groups,
        ) = self._video_objectives(fall_logits, metadata)
        if self.loss_weights.action > 0:
            action_values = F.cross_entropy(
                action_logits,
                action_targets,
                weight=self.action_class_weights.to(action_logits.dtype),
                reduction="none",
            )
            loss_action = _weighted_mean(action_values, action_weights)
        else:
            loss_action = action_logits.sum() * 0.0
        if self.loss_weights.control > 0:
            control_logits = outputs["control_logits"].float()
            controlled_ids = action_targets.new_tensor(
                CONTROLLED_DESCENT_ACTION_IDS
            )
            control_targets = action_targets[:, None].eq(
                controlled_ids[None]
            ).any(dim=1).to(dtype=control_logits.dtype)
            loss_control = _weighted_mean(
                _sigmoid_focal_loss(
                    control_logits,
                    control_targets,
                    alpha=self.control_alpha,
                    gamma=self.control_gamma,
                ),
                control_weights,
            )
        else:
            loss_control = fall_logits.sum() * 0.0
        loss = (
            self.loss_weights.stage * loss_stage
            + self.loss_weights.fall * loss_fall
            + self.loss_weights.event_state * loss_event_state
            + self.loss_weights.onset * loss_onset
            + self.loss_weights.action * loss_action
            + self.loss_weights.control * loss_control
            + self.loss_weights.geometry_stage * loss_geometry_stage
            + self.loss_weights.geometry_fall * loss_geometry_fall
            + self.loss_weights.consistency * loss_consistency
            + self.loss_weights.video_mil * loss_video_mil
            + self.loss_weights.video_ranking * loss_video_ranking
            + self.loss_weights.hard_video_ranking
            * loss_hard_video_ranking
        )
        supervised = event_weights > 0
        return {
            "loss": loss,
            "loss_stage": loss_stage,
            "loss_fall": loss_fall,
            "loss_event_state": loss_event_state,
            "loss_onset": loss_onset,
            "loss_action": loss_action,
            "loss_control": loss_control,
            "loss_geometry_stage": loss_geometry_stage,
            "loss_geometry_fall": loss_geometry_fall,
            "loss_consistency": loss_consistency,
            "loss_video_mil": loss_video_mil,
            "loss_video_ranking": loss_video_ranking,
            "loss_hard_video_ranking": loss_hard_video_ranking,
            "num_supervised": supervised.sum().detach(),
            "num_fall_positive": (
                supervised & fall_targets.gt(0.5)
            ).sum().detach(),
            "num_video_groups": num_video_groups.detach(),
            "num_video_positive_groups": num_video_positive_groups.detach(),
            "num_hard_negative_groups": num_hard_negative_groups.detach(),
        }
