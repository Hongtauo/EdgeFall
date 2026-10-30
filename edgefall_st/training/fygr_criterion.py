"""Loss wrapper for independent V3.4 global and ROI supervision."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn

from edgefall_st.training.bbox_criterion import (
    BBoxFallCriterion,
    _sigmoid_focal_loss,
    _weighted_mean,
)


class FYGRCriterion(nn.Module):
    def __init__(
        self,
        base: BBoxFallCriterion,
        *,
        global_aux_weight: float = 0.5,
        roi_aux_weight: float = 0.5,
        fall_alpha: float = 0.75,
        fall_gamma: float = 2.0,
        weak_video_aux_weight: float = 0.5,
        weak_video_topk: int = 2,
        fusion_hard_adl_ranking_weight: float = 0.0,
        fusion_hard_adl_ranking_margin: float = 0.2,
        fusion_hard_adl_categories: Sequence[str] = (),
        hard_negative_anchor_categories: Mapping[
            tuple[str, int], str
        ] | None = None,
        fusion_hard_tail_ranking_weight: float = 0.0,
        fusion_hard_tail_ranking_margin: float = 0.2,
        fusion_hard_tail_negative_topk: int = 3,
        fusion_hard_tail_positive_quantile: float = 0.25,
        fusion_hard_tail_categories: Sequence[str] = (),
        hard_negative_window_categories: Mapping[
            tuple[str, int], Sequence[str]
        ] | None = None,
    ) -> None:
        super().__init__()
        if min(
            global_aux_weight,
            roi_aux_weight,
            fall_gamma,
            weak_video_aux_weight,
            fusion_hard_adl_ranking_weight,
            fusion_hard_adl_ranking_margin,
            fusion_hard_tail_ranking_weight,
            fusion_hard_tail_ranking_margin,
        ) < 0:
            raise ValueError("Auxiliary weights/gamma cannot be negative")
        if weak_video_topk <= 0:
            raise ValueError("weak_video_topk must be positive")
        if fusion_hard_tail_negative_topk <= 0:
            raise ValueError("fusion_hard_tail_negative_topk must be positive")
        if not 0.0 < fusion_hard_tail_positive_quantile <= 1.0:
            raise ValueError(
                "fusion_hard_tail_positive_quantile must be in (0, 1]"
            )
        if not 0.0 <= fall_alpha <= 1.0:
            raise ValueError("fall_alpha must be in [0, 1]")
        self.base = base
        self.global_aux_weight = float(global_aux_weight)
        self.roi_aux_weight = float(roi_aux_weight)
        self.fall_alpha = float(fall_alpha)
        self.fall_gamma = float(fall_gamma)
        self.weak_video_aux_weight = float(weak_video_aux_weight)
        self.weak_video_topk = int(weak_video_topk)
        self.fusion_hard_adl_ranking_weight = float(
            fusion_hard_adl_ranking_weight
        )
        self.fusion_hard_adl_ranking_margin = float(
            fusion_hard_adl_ranking_margin
        )
        self.fusion_hard_adl_categories = frozenset(
            str(value) for value in fusion_hard_adl_categories
        )
        self.hard_negative_anchor_categories = dict(
            hard_negative_anchor_categories or {}
        )
        self.fusion_hard_tail_ranking_weight = float(
            fusion_hard_tail_ranking_weight
        )
        self.fusion_hard_tail_ranking_margin = float(
            fusion_hard_tail_ranking_margin
        )
        self.fusion_hard_tail_negative_topk = int(
            fusion_hard_tail_negative_topk
        )
        self.fusion_hard_tail_positive_quantile = float(
            fusion_hard_tail_positive_quantile
        )
        self.fusion_hard_tail_categories = frozenset(
            str(value) for value in fusion_hard_tail_categories
        )
        self.hard_negative_window_categories = {
            key: tuple(str(value) for value in values)
            for key, values in (hard_negative_window_categories or {}).items()
        }

    def _auxiliary_loss(
        self,
        logits: Tensor | None,
        targets: Sequence[dict[str, Tensor]],
    ) -> Tensor:
        if logits is None:
            reference = targets[0]["event_target"].float()
            return reference.sum() * 0.0
        fall_targets = torch.stack(
            [target["event_target"] for target in targets]
        ).to(dtype=logits.dtype)
        sample_weights = torch.stack(
            [
                target.get("event_loss_weight", target["action_loss_weight"])
                for target in targets
            ]
        ).to(dtype=logits.dtype)
        return _weighted_mean(
            _sigmoid_focal_loss(
                logits.float(),
                fall_targets.float(),
                alpha=self.fall_alpha,
                gamma=self.fall_gamma,
            ),
            sample_weights.float(),
        )

    def _weak_video_loss(
        self,
        logits: Tensor,
        metadata: object | None,
    ) -> Tensor:
        zero = logits.sum() * 0.0
        if (
            not isinstance(metadata, Sequence)
            or isinstance(metadata, (str, bytes))
            or len(metadata) != logits.shape[0]
        ):
            return zero
        grouped: dict[str, list[int]] = defaultdict(list)
        values: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for index, item in enumerate(metadata):
            if not isinstance(item, Mapping) or not bool(
                item.get("weak_video_label", False)
            ):
                continue
            path = str(item["path"])
            grouped[path].append(index)
            values[path].append(item)
        scores: list[Tensor] = []
        labels: list[float] = []
        for path, indices in grouped.items():
            items = values[path]
            label = bool(items[0]["video_has_fall_event"])
            if any(bool(item["video_has_fall_event"]) != label for item in items):
                raise ValueError(f"Inconsistent weak video target for {path!r}")
            index_tensor = torch.tensor(indices, device=logits.device)
            current = logits[index_tensor]
            if label:
                count = min(self.weak_video_topk, int(current.numel()))
                scores.append(current.topk(count, sorted=False).values.mean())
            else:
                scores.append(current.max())
            labels.append(float(label))
        if not scores:
            return zero
        score_tensor = torch.stack(scores)
        label_tensor = score_tensor.new_tensor(labels)
        return _sigmoid_focal_loss(
            score_tensor.float(),
            label_tensor.float(),
            alpha=0.5,
            gamma=self.fall_gamma,
        ).mean()

    def _fusion_hard_adl_ranking_loss(
        self,
        logits: Tensor,
        metadata: object | None,
    ) -> tuple[Tensor, int, int, int, int]:
        zero = logits.sum() * 0.0
        if (
            self.fusion_hard_adl_ranking_weight == 0.0
            or not self.hard_negative_anchor_categories
            or not self.fusion_hard_adl_categories
            or not isinstance(metadata, Sequence)
            or isinstance(metadata, (str, bytes))
            or len(metadata) != logits.shape[0]
        ):
            return zero, 0, 0, 0, 0

        positive_indices: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        hard_indices: dict[
            str, dict[str, dict[str, list[int]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        hard_anchor_count = 0
        for index, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                continue
            domain = str(item.get("domain", "omnifall"))
            path = str(item["path"])
            if bool(item.get("video_has_fall_event", False)):
                positive_indices[domain][path].append(index)
                continue
            key = (path, int(item.get("window_index", -1)))
            category = self.hard_negative_anchor_categories.get(key)
            if category not in self.fusion_hard_adl_categories:
                continue
            hard_indices[domain][str(category)][path].append(index)
            hard_anchor_count += 1

        domain_losses: list[Tensor] = []
        balanced_category_count = 0
        positive_video_count = 0
        for domain in sorted(set(positive_indices).intersection(hard_indices)):
            positive_scores: list[Tensor] = []
            for indices in positive_indices[domain].values():
                current = logits[
                    torch.tensor(indices, device=logits.device)
                ]
                count = min(self.weak_video_topk, int(current.numel()))
                positive_scores.append(
                    current.topk(count, sorted=False).values.mean()
                )
            if not positive_scores:
                continue
            positive_video_count += len(positive_scores)
            positives = torch.stack(positive_scores)
            category_losses: list[Tensor] = []
            for category in sorted(hard_indices[domain]):
                negative_scores = [
                    logits[
                        torch.tensor(indices, device=logits.device)
                    ].max()
                    for indices in hard_indices[domain][category].values()
                ]
                if not negative_scores:
                    continue
                negatives = torch.stack(negative_scores)
                category_losses.append(
                    torch.relu(
                        self.fusion_hard_adl_ranking_margin
                        - positives[:, None]
                        + negatives[None, :]
                    ).mean()
                )
            if category_losses:
                balanced_category_count += len(category_losses)
                domain_losses.append(torch.stack(category_losses).mean())
        if not domain_losses:
            return zero, 0, 0, positive_video_count, hard_anchor_count
        return (
            torch.stack(domain_losses).mean(),
            len(domain_losses),
            balanced_category_count,
            positive_video_count,
            hard_anchor_count,
        )

    def _fusion_hard_tail_ranking_loss(
        self,
        logits: Tensor,
        metadata: object | None,
    ) -> tuple[Tensor, int, int, int, int, int]:
        zero = logits.sum() * 0.0
        if (
            self.fusion_hard_tail_ranking_weight == 0.0
            or not self.hard_negative_window_categories
            or not self.fusion_hard_tail_categories
            or not isinstance(metadata, Sequence)
            or isinstance(metadata, (str, bytes))
            or len(metadata) != logits.shape[0]
        ):
            return zero, 0, 0, 0, 0, 0

        strong_positive_indices: dict[
            str, dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        weak_positive_indices: dict[
            str, dict[str, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        hard_indices: dict[
            str, dict[str, dict[str, list[int]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        negative_window_memberships = 0
        unknown_window_memberships = 0
        for index, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                continue
            domain = str(item.get("domain", "omnifall"))
            path = str(item["path"])
            if bool(item.get("video_has_fall_event", False)):
                if bool(item.get("weak_video_label", False)):
                    weak_positive_indices[domain][path].append(index)
                elif bool(item.get("event_target", False)):
                    strong_positive_indices[domain][path].append(index)
                continue
            key = (path, int(item.get("window_index", -1)))
            categories = self.hard_negative_window_categories.get(key, ())
            for category in categories:
                if category not in self.fusion_hard_tail_categories:
                    continue
                hard_indices[domain][category][path].append(index)
                negative_window_memberships += 1
                if category == "score_mined_unknown":
                    unknown_window_memberships += 1

        positive_scores_by_domain: dict[str, list[Tensor]] = defaultdict(list)
        for domain, paths in strong_positive_indices.items():
            for indices in paths.values():
                current = logits[torch.tensor(indices, device=logits.device)]
                count = max(
                    1,
                    min(
                        int(current.numel()),
                        math.ceil(
                            int(current.numel())
                            * self.fusion_hard_tail_positive_quantile
                        ),
                    ),
                )
                positive_scores_by_domain[domain].append(
                    current.topk(count, largest=False, sorted=False).values.mean()
                )
        for domain, paths in weak_positive_indices.items():
            for indices in paths.values():
                current = logits[torch.tensor(indices, device=logits.device)]
                count = min(self.weak_video_topk, int(current.numel()))
                positive_scores_by_domain[domain].append(
                    current.topk(count, sorted=False).values.mean()
                )

        domain_losses: list[Tensor] = []
        balanced_category_count = 0
        positive_video_count = 0
        for domain in sorted(
            set(positive_scores_by_domain).intersection(hard_indices)
        ):
            positive_values = positive_scores_by_domain[domain]
            if not positive_values:
                continue
            positives = torch.stack(positive_values)
            positive_video_count += len(positive_values)
            category_losses: list[Tensor] = []
            for category in sorted(hard_indices[domain]):
                negative_scores: list[Tensor] = []
                for indices in hard_indices[domain][category].values():
                    current = logits[
                        torch.tensor(indices, device=logits.device)
                    ]
                    count = min(
                        self.fusion_hard_tail_negative_topk,
                        int(current.numel()),
                    )
                    negative_scores.append(
                        current.topk(count, sorted=False).values.mean()
                    )
                if not negative_scores:
                    continue
                negatives = torch.stack(negative_scores)
                category_losses.append(
                    torch.relu(
                        self.fusion_hard_tail_ranking_margin
                        - positives[:, None]
                        + negatives[None, :]
                    ).mean()
                )
            if category_losses:
                balanced_category_count += len(category_losses)
                domain_losses.append(torch.stack(category_losses).mean())
        if not domain_losses:
            return (
                zero,
                0,
                0,
                positive_video_count,
                negative_window_memberships,
                unknown_window_memberships,
            )
        return (
            torch.stack(domain_losses).mean(),
            len(domain_losses),
            balanced_category_count,
            positive_video_count,
            negative_window_memberships,
            unknown_window_memberships,
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        metadata: object | None = None,
    ) -> dict[str, Tensor]:
        losses = self.base(outputs, targets, metadata)
        loss_global_aux = self._auxiliary_loss(
            outputs.get("global_fall_logits"), targets
        )
        loss_roi_aux = self._auxiliary_loss(
            outputs.get("roi_fall_logits"), targets
        )
        global_logits = outputs.get("global_fall_logits")
        roi_logits = outputs.get("roi_fall_logits")
        loss_global_aux_video_mil = (
            losses["loss"] * 0.0
            if global_logits is None
            else self._weak_video_loss(global_logits, metadata)
        )
        loss_roi_aux_video_mil = (
            losses["loss"] * 0.0
            if roi_logits is None
            else self._weak_video_loss(roi_logits, metadata)
        )
        (
            loss_fusion_hard_adl_ranking,
            ranking_domains,
            ranking_categories,
            ranking_positive_videos,
            ranking_hard_anchors,
        ) = self._fusion_hard_adl_ranking_loss(
            outputs["fall_logits"], metadata
        )
        (
            loss_fusion_hard_tail_ranking,
            hard_tail_domains,
            hard_tail_categories,
            hard_tail_positive_videos,
            hard_tail_negative_windows,
            hard_tail_unknown_windows,
        ) = self._fusion_hard_tail_ranking_loss(
            outputs["fall_logits"], metadata
        )
        losses["loss_global_aux"] = loss_global_aux
        losses["loss_roi_aux"] = loss_roi_aux
        losses["loss_global_aux_video_mil"] = loss_global_aux_video_mil
        losses["loss_roi_aux_video_mil"] = loss_roi_aux_video_mil
        losses["loss_fusion_hard_adl_ranking"] = (
            loss_fusion_hard_adl_ranking
        )
        losses["loss_fusion_hard_tail_ranking"] = (
            loss_fusion_hard_tail_ranking
        )
        losses["num_fusion_ranking_domains"] = losses["loss"].new_tensor(
            ranking_domains
        )
        losses["num_fusion_ranking_categories"] = losses["loss"].new_tensor(
            ranking_categories
        )
        losses["num_fusion_ranking_positive_videos"] = losses[
            "loss"
        ].new_tensor(ranking_positive_videos)
        losses["num_fusion_ranking_hard_anchors"] = losses["loss"].new_tensor(
            ranking_hard_anchors
        )
        losses["num_fusion_hard_tail_domains"] = losses["loss"].new_tensor(
            hard_tail_domains
        )
        losses["num_fusion_hard_tail_categories"] = losses["loss"].new_tensor(
            hard_tail_categories
        )
        losses["num_fusion_hard_tail_positive_videos"] = losses[
            "loss"
        ].new_tensor(hard_tail_positive_videos)
        losses["num_fusion_hard_tail_negative_windows"] = losses[
            "loss"
        ].new_tensor(hard_tail_negative_windows)
        losses["num_fusion_hard_tail_unknown_windows"] = losses[
            "loss"
        ].new_tensor(hard_tail_unknown_windows)
        losses["loss"] = (
            losses["loss"]
            + self.global_aux_weight
            * (
                loss_global_aux
                + self.weak_video_aux_weight * loss_global_aux_video_mil
            )
            + self.roi_aux_weight
            * (
                loss_roi_aux
                + self.weak_video_aux_weight * loss_roi_aux_video_mil
            )
            + self.fusion_hard_adl_ranking_weight
            * loss_fusion_hard_adl_ranking
            + self.fusion_hard_tail_ranking_weight
            * loss_fusion_hard_tail_ranking
        )
        return losses
