"""Train the V3.4 frozen-YOLO global/ROI visual confirmer."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from edgefall_st.evaluation.video_metrics import (
    end_to_end_video_metrics,
)
from edgefall_st.data.labels import ACTION_LABELS
from edgefall_st.features.cache import (
    cache_person_capacity,
    FrozenFeatureWindowDataset,
    MultiDomainFrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
    roi_feature_shape,
)
from edgefall_st.models.fygr import (
    FYGR_FUSION_CONCAT_MLP,
    FYGR_FUSION_CROSS_ATTENTION,
    FYGR_FUSION_MODES,
    FYGR_FUSION_TEMPORAL_TOKENS,
    FrozenYoloGlobalROIConfirmer,
)
from edgefall_st.training.bbox_criterion import (
    BBoxClassifierLossWeights,
    BBoxFallCriterion,
)
from edgefall_st.training.fygr_criterion import FYGRCriterion
from edgefall_st.training.hard_negatives import (
    HardNegativeRegistry,
    empty_hard_negative_registry,
    load_hard_negative_registry,
)
from edgefall_st.training.hard_positives import (
    HardPositiveRegistry,
    empty_hard_positive_registry,
    load_hard_positive_registry,
)
from edgefall_st.training.sampler import VideoGroupedBatchSampler
from edgefall_st.training.train import (
    _ClassifierMetricAccumulator,
    _MetricAccumulator,
    _format_metrics,
    _move_targets,
    _truncate_history_mask,
)


DEFAULT_FUSION_HARD_ADL_CATEGORIES = (
    "lie_down",
    "lying",
    "sit",
    "sit_down",
    "kneel_squat",
    "squat",
    "bend_pick",
)
DEFAULT_FUSION_HARD_TAIL_CATEGORIES = (
    *DEFAULT_FUSION_HARD_ADL_CATEGORIES,
    "score_mined_unknown",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-cache",
        type=Path,
        default=Path("artifacts/frozen_features/yolo26x/train"),
    )
    data.add_argument(
        "--val-cache",
        type=Path,
        default=Path("artifacts/frozen_features/yolo26x/val"),
    )
    data.add_argument("--shard-cache-size", type=int, default=2)
    data.add_argument("--train-candidates", type=Path)
    data.add_argument("--hard-negative-manifest", type=Path)
    data.add_argument("--hard-positive-manifest", type=Path)
    data.add_argument("--split-manifest", type=Path)
    data.add_argument(
        "--extra-train-cache", type=Path, action="append", default=[]
    )
    data.add_argument(
        "--extra-val-cache", type=Path, action="append", default=[]
    )
    data.add_argument(
        "--train-overlay-cache",
        type=Path,
        action="append",
        default=[],
        help="Train-only cache replacing paths removed from the primary cache",
    )
    data.add_argument("--workers", type=int, default=2)
    data.add_argument("--prefetch-factor", type=int, default=2)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--geometry-checkpoint",
        type=Path,
        default=Path(
            "artifacts/checkpoints/20260725_220646_947213/best.pt"
        ),
    )
    model.add_argument("--temporal-dim", type=int, default=128)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument(
        "--fusion-mode",
        choices=FYGR_FUSION_MODES,
        default=FYGR_FUSION_CONCAT_MLP,
        help="V3.5 concat baseline or V3.6 geometry-conditioned visual fusion",
    )
    model.add_argument("--attention-heads", type=int, default=4)
    model.add_argument("--attention-ffn-dim", type=int, default=256)
    model.add_argument(
        "--person-mil-temperature",
        type=float,
        default=0.0,
        help="0 uses hard max; positive values use normalized LogMeanExp",
    )
    model.add_argument(
        "--use-global", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--use-roi", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--train-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Jointly fine-tune the pretrained V3.3-G geometry branch",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=15)
    training.add_argument("--batch-size", type=int, default=72)
    training.add_argument("--windows-per-video", type=int, default=12)
    training.add_argument("--positive-videos-per-batch", type=int, default=2)
    training.add_argument(
        "--hard-positive-videos-per-batch", type=int, default=0
    )
    training.add_argument(
        "--semantic-hard-videos-per-batch", type=int, default=0
    )
    training.add_argument(
        "--score-hard-videos-per-batch", type=int, default=0
    )
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument(
        "--geometry-lr-factor",
        type=float,
        default=0.1,
        help="Geometry learning rate relative to --learning-rate",
    )
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--warmup-epochs", type=int, default=2)
    training.add_argument("--minimum-lr-ratio", type=float, default=0.05)
    training.add_argument("--gradient-clip", type=float, default=10.0)
    training.add_argument("--history-truncation-prob", type=float, default=0.2)
    training.add_argument("--seed", type=int, default=20260726)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    training.add_argument("--resume", type=Path)
    training.add_argument("--max-train-batches", type=int, default=0)
    training.add_argument("--max-val-batches", type=int, default=0)
    training.add_argument("--log-interval", type=int, default=50)

    losses = parser.add_argument_group("losses")
    losses.add_argument("--stage-loss-weight", type=float, default=1.0)
    losses.add_argument("--fall-loss-weight", type=float, default=2.0)
    losses.add_argument("--event-state-loss-weight", type=float, default=0.5)
    losses.add_argument("--onset-loss-weight", type=float, default=0.5)
    losses.add_argument("--action-loss-weight", type=float, default=0.05)
    losses.add_argument("--control-loss-weight", type=float, default=0.25)
    losses.add_argument("--consistency-loss-weight", type=float, default=0.25)
    losses.add_argument("--video-mil-loss-weight", type=float, default=0.5)
    losses.add_argument("--video-ranking-loss-weight", type=float, default=0.25)
    losses.add_argument(
        "--hard-video-ranking-loss-weight", type=float, default=0.0
    )
    losses.add_argument("--hard-video-ranking-margin", type=float, default=0.2)
    losses.add_argument(
        "--fusion-hard-adl-ranking-loss-weight", type=float, default=0.0
    )
    losses.add_argument(
        "--fusion-hard-adl-ranking-margin", type=float, default=0.2
    )
    losses.add_argument(
        "--fusion-hard-adl-category",
        action="append",
        default=None,
        help=(
            "Semantic hard-negative category used by domain-balanced fusion "
            "ranking; repeat to override the default category set"
        ),
    )
    losses.add_argument(
        "--fusion-hard-tail-ranking-loss-weight", type=float, default=0.0
    )
    losses.add_argument(
        "--fusion-hard-tail-ranking-margin", type=float, default=0.2
    )
    losses.add_argument(
        "--fusion-hard-tail-negative-topk", type=int, default=3
    )
    losses.add_argument(
        "--fusion-hard-tail-positive-quantile", type=float, default=0.25
    )
    losses.add_argument(
        "--fusion-hard-tail-category",
        action="append",
        default=None,
        help=(
            "Hard-negative chunk category used by V3.10 tail ranking; "
            "repeat to override semantic plus score-mined defaults"
        ),
    )
    losses.add_argument("--global-aux-weight", type=float, default=0.5)
    losses.add_argument("--roi-aux-weight", type=float, default=0.5)
    losses.add_argument("--weak-video-aux-weight", type=float, default=0.5)
    losses.add_argument("--fall-alpha", type=float, default=0.75)
    losses.add_argument("--fall-gamma", type=float, default=2.0)

    output = parser.add_argument_group("output")
    output.add_argument("--log-root", type=Path, default=Path("log"))
    output.add_argument(
        "--checkpoint-root", type=Path, default=Path("artifacts/checkpoints")
    )
    return parser


def _fusion_hard_adl_categories(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    values = getattr(args, "fusion_hard_adl_category", None)
    if values is None:
        return DEFAULT_FUSION_HARD_ADL_CATEGORIES
    return tuple(str(value).strip() for value in values)


def _fusion_hard_tail_categories(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    values = getattr(args, "fusion_hard_tail_category", None)
    if values is None:
        return DEFAULT_FUSION_HARD_TAIL_CATEGORIES
    return tuple(str(value).strip() for value in values)


def _validate(args: argparse.Namespace) -> None:
    positive = (
        "shard_cache_size",
        "prefetch_factor",
        "epochs",
        "batch_size",
        "windows_per_video",
        "temporal_dim",
        "attention_heads",
        "attention_ffn_dim",
        "fusion_hard_tail_negative_topk",
        "log_interval",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.warmup_epochs < 0:
        raise ValueError("workers/warmup-epochs cannot be negative")
    if args.batch_size % args.windows_per_video:
        raise ValueError("batch-size must be divisible by windows-per-video")
    videos_per_batch = args.batch_size // args.windows_per_video
    if not 0 <= args.positive_videos_per_batch < videos_per_batch:
        raise ValueError("positive-videos-per-batch is out of range")
    if not 0 <= args.hard_positive_videos_per_batch <= args.positive_videos_per_batch:
        raise ValueError("hard-positive-videos-per-batch is out of range")
    if (
        args.hard_positive_videos_per_batch > 0
        and args.hard_positive_manifest is None
    ):
        raise ValueError(
            "Hard-positive sampling requires --hard-positive-manifest"
        )
    hard_slots = (
        args.semantic_hard_videos_per_batch
        + args.score_hard_videos_per_batch
    )
    if min(
        args.semantic_hard_videos_per_batch,
        args.score_hard_videos_per_batch,
    ) < 0:
        raise ValueError("Hard-negative video slots cannot be negative")
    if hard_slots > videos_per_batch - args.positive_videos_per_batch:
        raise ValueError(
            "Hard-negative video slots exceed negative batch capacity"
        )
    if (
        hard_slots > 0
        or args.hard_video_ranking_loss_weight > 0
        or args.fusion_hard_adl_ranking_loss_weight > 0
        or args.fusion_hard_tail_ranking_loss_weight > 0
    ) and args.hard_negative_manifest is None:
        raise ValueError(
            "Hard-negative sampling/loss requires --hard-negative-manifest"
        )
    if not args.use_global and not args.use_roi:
        raise ValueError("At least one of --use-global/--use-roi is required")
    if (
        args.fusion_mode
        in (FYGR_FUSION_CROSS_ATTENTION, FYGR_FUSION_TEMPORAL_TOKENS)
        and args.temporal_dim % args.attention_heads
    ):
        raise ValueError(
            "temporal-dim must be divisible by attention-heads"
        )
    for name in (
        "dropout",
        "history_truncation_prob",
        "minimum_lr_ratio",
        "fall_alpha",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")
    if min(args.learning_rate, args.weight_decay, args.gradient_clip) < 0:
        raise ValueError("Optimizer values cannot be negative")
    if args.person_mil_temperature < 0:
        raise ValueError("person-mil-temperature cannot be negative")
    if not 0.0 < args.geometry_lr_factor <= 1.0:
        raise ValueError("geometry-lr-factor must be in (0, 1]")
    if min(
        args.weak_video_aux_weight,
        args.hard_video_ranking_loss_weight,
        args.hard_video_ranking_margin,
        args.fusion_hard_adl_ranking_loss_weight,
        args.fusion_hard_adl_ranking_margin,
        args.fusion_hard_tail_ranking_loss_weight,
        args.fusion_hard_tail_ranking_margin,
    ) < 0:
        raise ValueError("Auxiliary loss weights/margins cannot be negative")
    if not 0.0 < args.fusion_hard_tail_positive_quantile <= 1.0:
        raise ValueError(
            "fusion-hard-tail-positive-quantile must be in (0, 1]"
        )
    if (
        args.fusion_hard_adl_ranking_loss_weight > 0
        and args.fusion_hard_tail_ranking_loss_weight > 0
    ):
        raise ValueError(
            "Single-anchor and hard-tail fusion ranking cannot both be enabled"
        )
    categories = _fusion_hard_adl_categories(args)
    if any(not value.strip() for value in categories):
        raise ValueError("Fusion hard-ADL categories cannot be empty")
    if len(set(categories)) != len(categories):
        raise ValueError("Fusion hard-ADL categories must be unique")
    tail_categories = _fusion_hard_tail_categories(args)
    if any(not value.strip() for value in tail_categories):
        raise ValueError("Fusion hard-tail categories cannot be empty")
    if len(set(tail_categories)) != len(tail_categories):
        raise ValueError("Fusion hard-tail categories must be unique")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        raise ValueError("Maximum batch counts cannot be negative")
    if args.split_manifest is not None and args.max_val_batches:
        raise ValueError(
            "Manifest-based model selection requires complete validation; "
            "max-val-batches must be zero"
        )
    if len(args.extra_train_cache) != len(args.extra_val_cache):
        raise ValueError("Extra train/val cache counts must match")
    if args.extra_train_cache and args.split_manifest is None:
        raise ValueError("Multi-domain training requires --split-manifest")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compatible_manifests(train: dict[str, Any], val: dict[str, Any]) -> None:
    keys = (
        "format",
        "history_offsets",
        "visual_offsets",
        "visual_input_size",
        "geometry_input_size",
        "weights_sha256",
        "feature_adapter",
    )
    mismatched = [key for key in keys if train.get(key) != val.get(key)]
    train_arrays = train.get("arrays", {})
    val_arrays = val.get("arrays", {})
    if train_arrays.get("global_features") != val_arrays.get("global_features"):
        mismatched.append("arrays.global_features")
    if roi_feature_shape(train) != roi_feature_shape(val):
        mismatched.append("arrays.roi_features")
    if mismatched:
        raise ValueError(
            "Train/val feature cache contracts differ: " + ", ".join(mismatched)
        )


def _candidate_indices(path: Path) -> list[int]:
    selected: list[int] = []
    previous = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "cache_index" not in value:
                raise ValueError(
                    f"Invalid candidate row at {path}:{line_number}"
                )
            index = int(value["cache_index"])
            if index <= previous:
                raise ValueError("Candidate cache indices must be increasing")
            previous = index
            if bool(value.get("selected", False)):
                selected.append(index)
    if not selected:
        raise ValueError("Candidate manifest selected no training windows")
    return selected


def _selection_paths(
    path: Path | None,
) -> dict[tuple[str, str], set[str]]:
    if path is None:
        return {}
    selected: dict[tuple[str, str], set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["domain"]), str(row["split"]))
            selected.setdefault(key, set()).add(str(row["path"]))
    return selected


def _argument_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_argument_value(item) for item in value]
    return value


def _cache_dataset(
    root: Path,
    *,
    split: str,
    selections: dict[tuple[str, str], set[str]],
    shard_cache_size: int,
    selected_indices: Sequence[int] | None = None,
    default_domain: str = "omnifall",
) -> FrozenFeatureWindowDataset:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    domain = str(manifest.get("domain", default_domain))
    selected_paths = selections.get((domain, split))
    if selections and selected_paths is None:
        raise ValueError(
            f"Split manifest has no {domain}/{split} selection for {root}"
        )
    return FrozenFeatureWindowDataset(
        root,
        shard_cache_size=shard_cache_size,
        selected_indices=selected_indices,
        selected_paths=selected_paths,
        domain=domain,
    )


def _combine_datasets(
    datasets: Sequence[FrozenFeatureWindowDataset],
) -> FrozenFeatureWindowDataset | MultiDomainFrozenFeatureWindowDataset:
    if len(datasets) == 1:
        return datasets[0]
    reference = datasets[0].manifest
    for dataset in datasets[1:]:
        _compatible_manifests(reference, dataset.manifest)
    return MultiDomainFrozenFeatureWindowDataset(datasets)


def _logger(args: argparse.Namespace, device: torch.device):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = args.log_root / run_id
    checkpoint_dir = args.checkpoint_root / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"edgefall_st.fygr.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    payload = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": {
            key: _argument_value(value)
            for key, value in vars(args).items()
        },
        "resolved": {
            "device": str(device),
            "torch_version": str(torch.__version__),
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }
    with (log_dir / "parameters.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return run_id, log_dir, checkpoint_dir, logger


def _update_parameters(
    log_dir: Path,
    *,
    feature_contract: dict[str, Any],
    model: FrozenYoloGlobalROIConfirmer,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    hard_negative_summary: dict[str, Any] | None = None,
    hard_positive_summary: dict[str, Any] | None = None,
) -> None:
    path = log_dir / "parameters.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["resolved"].update(
        {
            "feature_contract": feature_contract,
            "total_parameters": model.parameter_count,
            "trainable_parameters": model.trainable_parameter_count,
            "geometry_frozen": model.freeze_geometry,
            "geometry_trainable_parameters": sum(
                parameter.numel()
                for parameter in model.geometry.parameters()
                if parameter.requires_grad
            ),
            "amp_enabled": amp_enabled,
            "amp_dtype": str(amp_dtype),
            "parameter_scope": (
                ("frozen" if model.freeze_geometry else "trainable")
                + " V3.3-G plus trainable confirmer; YOLO feature extractor "
                "excluded"
            ),
            "hard_negative_summary": hard_negative_summary or {},
            "hard_positive_summary": hard_positive_summary or {},
        }
    )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _make_loaders(
    args: argparse.Namespace,
    train: FrozenFeatureWindowDataset | MultiDomainFrozenFeatureWindowDataset,
    val: FrozenFeatureWindowDataset | MultiDomainFrozenFeatureWindowDataset,
    hard_negatives: HardNegativeRegistry | None = None,
    hard_positives: HardPositiveRegistry | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], VideoGroupedBatchSampler]:
    registry = hard_negatives or empty_hard_negative_registry()
    positive_registry = hard_positives or empty_hard_positive_registry()
    sampler = VideoGroupedBatchSampler(
        train.windows,
        batch_size=args.batch_size,
        windows_per_video=args.windows_per_video,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
        positive_paths={
            path for path, value in train.video_has_fall_event.items() if value
        },
        weak_positive_paths=getattr(train, "weak_positive_paths", set()),
        positive_videos_per_batch=args.positive_videos_per_batch,
        event_stage_balanced=True,
        hard_positive_anchors=dict(positive_registry.anchors),
        hard_positive_anchor_domains=dict(positive_registry.anchor_domains),
        hard_positive_videos_per_batch=args.hard_positive_videos_per_batch,
        semantic_hard_anchors=dict(registry.semantic_anchors),
        score_hard_anchors=dict(registry.score_anchors),
        hard_anchor_categories=dict(registry.anchor_categories),
        semantic_hard_videos_per_batch=(
            args.semantic_hard_videos_per_batch
        ),
        score_hard_videos_per_batch=args.score_hard_videos_per_batch,
    )
    common: dict[str, Any] = {
        "num_workers": args.workers,
        "collate_fn": frozen_feature_collate_fn,
        "pin_memory": True,
    }
    if args.workers > 0:
        common.update(
            prefetch_factor=args.prefetch_factor, persistent_workers=True
        )
    train_loader = DataLoader(train, batch_sampler=sampler, **common)
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, sampler


def _criterion(
    args: argparse.Namespace,
    hard_negatives: HardNegativeRegistry | None = None,
) -> FYGRCriterion:
    registry = hard_negatives or empty_hard_negative_registry()
    base = BBoxFallCriterion(
        loss_weights=BBoxClassifierLossWeights(
            stage=args.stage_loss_weight,
            fall=args.fall_loss_weight,
            event_state=args.event_state_loss_weight,
            onset=args.onset_loss_weight,
            action=args.action_loss_weight,
            control=args.control_loss_weight,
            consistency=args.consistency_loss_weight,
            video_mil=args.video_mil_loss_weight,
            video_ranking=args.video_ranking_loss_weight,
            hard_video_ranking=float(
                getattr(args, "hard_video_ranking_loss_weight", 0.0)
            ),
        ),
        fall_alpha=args.fall_alpha,
        fall_gamma=args.fall_gamma,
        hard_video_ranking_margin=float(
            getattr(args, "hard_video_ranking_margin", 0.2)
        ),
        hard_negative_window_keys=registry.window_keys,
        video_pooling="asymmetric",
        video_topk=2,
    )
    return FYGRCriterion(
        base,
        global_aux_weight=args.global_aux_weight if args.use_global else 0.0,
        roi_aux_weight=args.roi_aux_weight if args.use_roi else 0.0,
        fall_alpha=args.fall_alpha,
        fall_gamma=args.fall_gamma,
        weak_video_aux_weight=args.weak_video_aux_weight,
        fusion_hard_adl_ranking_weight=float(
            getattr(args, "fusion_hard_adl_ranking_loss_weight", 0.0)
        ),
        fusion_hard_adl_ranking_margin=float(
            getattr(args, "fusion_hard_adl_ranking_margin", 0.2)
        ),
        fusion_hard_adl_categories=_fusion_hard_adl_categories(args),
        hard_negative_anchor_categories=(
            registry.anchor_window_categories
        ),
        fusion_hard_tail_ranking_weight=float(
            getattr(args, "fusion_hard_tail_ranking_loss_weight", 0.0)
        ),
        fusion_hard_tail_ranking_margin=float(
            getattr(args, "fusion_hard_tail_ranking_margin", 0.2)
        ),
        fusion_hard_tail_negative_topk=int(
            getattr(args, "fusion_hard_tail_negative_topk", 3)
        ),
        fusion_hard_tail_positive_quantile=float(
            getattr(args, "fusion_hard_tail_positive_quantile", 0.25)
        ),
        fusion_hard_tail_categories=_fusion_hard_tail_categories(args),
        hard_negative_window_categories=registry.window_categories,
    )


def _scheduler(args: argparse.Namespace, optimizer: AdamW) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if args.warmup_epochs and epoch < args.warmup_epochs:
            return max((epoch + 1) / args.warmup_epochs, 1e-3)
        span = max(1, args.epochs - args.warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - args.warmup_epochs) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.minimum_lr_ratio + (1.0 - args.minimum_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _optimizer(
    args: argparse.Namespace, model: FrozenYoloGlobalROIConfirmer
) -> AdamW:
    confirmer_parameters = []
    geometry_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("geometry."):
            geometry_parameters.append(parameter)
        else:
            confirmer_parameters.append(parameter)
    if not confirmer_parameters:
        raise RuntimeError("No trainable confirmer parameters were found")
    groups: list[dict[str, Any]] = [
        {
            "params": confirmer_parameters,
            "lr": args.learning_rate,
            "group_name": "confirmer",
        }
    ]
    if args.train_geometry:
        if not geometry_parameters:
            raise RuntimeError("Geometry training was requested but remains frozen")
        groups.append(
            {
                "params": geometry_parameters,
                "lr": args.learning_rate * args.geometry_lr_factor,
                "group_name": "geometry",
            }
        )
    elif geometry_parameters:
        raise RuntimeError("Geometry parameters are trainable without --train-geometry")
    return AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)


def _batch_count(loader: DataLoader[Any], maximum: int) -> int:
    return min(len(loader), maximum) if maximum else len(loader)


def _slice_outputs(
    outputs: dict[str, torch.Tensor], indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    index_tensor = torch.tensor(indices, device=next(iter(outputs.values())).device)
    return {
        name: value[index_tensor]
        if value.ndim > 0 and value.shape[0] >= len(indices)
        else value
        for name, value in outputs.items()
    }


class _RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.squared_total = 0.0

    def update(self, values: torch.Tensor) -> None:
        finite = values.detach().float().reshape(-1).cpu()
        finite = finite[torch.isfinite(finite)]
        if not finite.numel():
            return
        self.count += int(finite.numel())
        self.total += float(finite.sum())
        self.squared_total += float((finite * finite).sum())

    def result(self, prefix: str) -> dict[str, float]:
        if not self.count:
            return {}
        mean = self.total / self.count
        variance = max(0.0, self.squared_total / self.count - mean * mean)
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
        }


class _FusionStabilityAccumulator:
    def __init__(
        self, *, use_global: bool, use_roi: bool, visual_slots: int
    ) -> None:
        self.use_global = bool(use_global)
        self.use_roi = bool(use_roi)
        self.visual_slots = int(visual_slots)
        self.state_norm = _RunningMoments()
        self.attention_entropy = _RunningMoments()
        self.attention_peak = _RunningMoments()
        self.global_mass = _RunningMoments()
        self.roi_mass = _RunningMoments()

    def update(self, outputs: Mapping[str, torch.Tensor]) -> None:
        state = outputs.get(
            "fusion_pre_norm_state", outputs.get("fusion_state")
        )
        if state is not None:
            self.state_norm.update(state.norm(dim=-1))
        weights = outputs.get("fusion_attention_weights")
        if weights is None:
            return
        if weights.ndim == 3:
            weights = weights[:, -1]
        if weights.ndim != 2:
            raise ValueError("Unexpected fusion attention weight shape")
        key_valid = outputs.get("fusion_key_valid_mask")
        if key_valid is None:
            key_valid = torch.ones_like(weights, dtype=torch.bool)
        else:
            key_valid = key_valid.bool()
        if key_valid.shape != weights.shape:
            raise ValueError("Fusion attention weights and key mask disagree")
        probabilities = weights.float() * key_valid.to(weights.dtype)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        entropy = -(
            probabilities
            * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        valid_count = key_valid.sum(dim=-1)
        normalized_entropy = torch.where(
            valid_count > 1,
            entropy / valid_count.float().log().clamp_min(1e-12),
            torch.zeros_like(entropy),
        )
        self.attention_entropy.update(normalized_entropy)
        self.attention_peak.update(probabilities.max(dim=-1).values)
        offset = 0
        slots = self.visual_slots if weights.shape[1] > 2 else 1
        if self.use_global:
            self.global_mass.update(
                probabilities[:, offset : offset + slots].sum(dim=-1)
            )
            offset += slots
        if self.use_roi:
            self.roi_mass.update(
                probabilities[:, offset : offset + slots].sum(dim=-1)
            )

    def result(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for prefix, meter in (
            ("fusion_state_norm", self.state_norm),
            ("fusion_attention_entropy", self.attention_entropy),
            ("fusion_attention_peak", self.attention_peak),
            ("fusion_global_mass", self.global_mass),
            ("fusion_roi_mass", self.roi_mass),
        ):
            result.update(meter.result(prefix))
        return result


class _HardTailMetricAccumulator:
    """Track video-score tails by domain and peak negative action."""

    def __init__(self) -> None:
        self.video_scores: dict[tuple[str, str], float] = {}
        self.video_targets: dict[tuple[str, str], bool] = {}
        self.peak_actions: dict[tuple[str, str], str] = {}

    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        scores = outputs["fall_logits"].detach().float().sigmoid().cpu()
        for index, item in enumerate(metadata):
            domain = str(item.get("domain", "omnifall"))
            path = str(item["path"])
            key = (domain, path)
            target = bool(item.get("video_has_fall_event", False))
            previous_target = self.video_targets.get(key)
            if previous_target is not None and previous_target != target:
                raise ValueError(f"Inconsistent video target for {path!r}")
            self.video_targets[key] = target
            score = float(scores[index])
            if score < self.video_scores.get(key, -math.inf):
                continue
            self.video_scores[key] = score
            action_id = int(
                item.get("window_action_label", item.get("action_label", 9))
            )
            self.peak_actions[key] = ACTION_LABELS.get(
                action_id, f"action_{action_id}"
            )

    @staticmethod
    def _quantile(values: Sequence[float], probability: float) -> float:
        if not values:
            return 0.0
        return float(np.quantile(np.asarray(values, dtype=np.float64), probability))

    def result(self) -> dict[str, float]:
        result: dict[str, float] = {}
        domains = sorted({key[0] for key in self.video_scores})
        for domain in domains:
            keys = [key for key in self.video_scores if key[0] == domain]
            positives = [
                self.video_scores[key]
                for key in keys
                if self.video_targets[key]
            ]
            negatives = [
                self.video_scores[key]
                for key in keys
                if not self.video_targets[key]
            ]
            prefix = f"{domain}_tail"
            result[f"{prefix}_positive_count"] = float(len(positives))
            result[f"{prefix}_negative_count"] = float(len(negatives))
            for name, probability in (("p05", 0.05), ("p10", 0.10)):
                result[f"{prefix}_positive_score_{name}"] = self._quantile(
                    positives, probability
                )
            for name, probability in (
                ("p90", 0.90),
                ("p95", 0.95),
                ("p99", 0.99),
            ):
                result[f"{prefix}_negative_score_{name}"] = self._quantile(
                    negatives, probability
                )
            grouped: dict[str, list[float]] = {}
            for key in keys:
                if self.video_targets[key]:
                    continue
                grouped.setdefault(self.peak_actions[key], []).append(
                    self.video_scores[key]
                )
            for action, values in sorted(grouped.items()):
                action_prefix = f"{prefix}_negative_{action}"
                result[f"{action_prefix}_count"] = float(len(values))
                result[f"{action_prefix}_score_mean"] = sum(values) / len(values)
                for name, probability in (
                    ("p90", 0.90),
                    ("p95", 0.95),
                    ("p99", 0.99),
                ):
                    result[f"{action_prefix}_score_{name}"] = self._quantile(
                        values, probability
                    )
        return result


def run_epoch(
    model: FrozenYoloGlobalROIConfirmer,
    criterion: FYGRCriterion,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gradient_clip: float,
    history_truncation_prob: float,
    max_batches: int,
    log_interval: int,
    logger: logging.Logger,
    epoch: int,
    phase: str,
    video_manifest: Path | None = None,
    manifest_split: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter = _MetricAccumulator()
    classifier_meter = _ClassifierMetricAccumulator()
    fusion_meter = _FusionStabilityAccumulator(
        use_global=model.use_global,
        use_roi=model.use_roi,
        visual_slots=len(model.visual_offsets),
    )
    hard_tail_meter = _HardTailMetricAccumulator()
    domain_meters: dict[str, _ClassifierMetricAccumulator] = {}
    domain_fusion_meters: dict[str, _FusionStabilityAccumulator] = {}
    manifest_records: dict[str, list[dict[str, Any]]] = {}
    if video_manifest is not None and (training or manifest_split is None):
        raise ValueError(
            "Manifest video metrics require evaluation and an explicit split"
        )
    total_batches = _batch_count(loader, max_batches)
    start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        global_features = batch["global_features"].to(
            device, non_blocking=True
        )
        roi_features = batch["roi_features"].to(device, non_blocking=True)
        actor_boxes = batch["actor_boxes"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        actor_valid_mask = batch["actor_valid_mask"].to(
            device, non_blocking=True
        )
        if device.type == "cpu":
            global_features = global_features.float()
            roi_features = roi_features.float()
        if training:
            valid_mask = _truncate_history_mask(
                valid_mask, history_truncation_prob
            )
            optimizer.zero_grad(set_to_none=True)
        targets = _move_targets(batch["targets"], device)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context, torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(
                global_features,
                roi_features,
                actor_boxes,
                valid_mask,
                actor_valid_mask,
            )
            losses = criterion(outputs, targets, batch["metadata"])
        if training:
            scaler.scale(losses["loss"]).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    gradient_clip,
                )
            scaler.step(optimizer)
            scaler.update()
        count = len(targets)
        meter.update(losses, count)
        classifier_meter.update(outputs, targets, batch["metadata"])
        fusion_meter.update(outputs)
        hard_tail_meter.update(outputs, batch["metadata"])
        if video_manifest is not None:
            fall_scores = outputs["fall_logits"].detach().float().sigmoid().cpu()
            for sample_index, metadata in enumerate(batch["metadata"]):
                domain = str(metadata.get("domain", "omnifall"))
                manifest_records.setdefault(domain, []).append(
                    {
                        "path": str(metadata["path"]),
                        "score": float(fall_scores[sample_index]),
                    }
                )
        domain_indices: dict[str, list[int]] = {}
        for sample_index, metadata in enumerate(batch["metadata"]):
            domain_indices.setdefault(
                str(metadata.get("domain", "omnifall")), []
            ).append(sample_index)
        for domain, indices in domain_indices.items():
            sliced_outputs = _slice_outputs(outputs, indices)
            domain_meters.setdefault(
                domain, _ClassifierMetricAccumulator()
            ).update(
                sliced_outputs,
                [targets[index] for index in indices],
                [batch["metadata"][index] for index in indices],
            )
            domain_fusion_meters.setdefault(
                domain,
                _FusionStabilityAccumulator(
                    use_global=model.use_global,
                    use_roi=model.use_roi,
                    visual_slots=len(model.visual_offsets),
                ),
            ).update(sliced_outputs)
        if (batch_index + 1) % log_interval == 0 or batch_index + 1 == total_batches:
            elapsed = time.perf_counter() - start
            logger.info(
                "epoch=%d %s_batch=%d/%d %s samples_per_s=%.1f",
                epoch,
                phase,
                batch_index + 1,
                total_batches,
                _format_metrics(meter.result()),
                meter.sample_count / max(elapsed, 1e-6),
            )
    result = meter.result()
    result.update(classifier_meter.result())
    result.update(fusion_meter.result())
    result.update(hard_tail_meter.result())
    domain_values: list[float] = []
    for domain, domain_meter in sorted(domain_meters.items()):
        current = domain_meter.result()
        result.update(
            {f"{domain}_{name}": value for name, value in current.items()}
        )
        result.update(
            {
                f"{domain}_{name}": value
                for name, value in domain_fusion_meters[domain]
                .result()
                .items()
            }
        )
        if "video_mean_precision" in current:
            domain_values.append(float(current["video_mean_precision"]))
    if domain_values:
        result["macro_video_mean_precision"] = sum(domain_values) / len(
            domain_values
        )
    if video_manifest is not None:
        end_to_end_values: list[float] = []
        domains = tuple(sorted(manifest_records))
        if not domains:
            raise ValueError("Manifest validation produced no video records")
        last_metrics: dict[str, Any] = {}
        for domain in domains:
            current, _ = end_to_end_video_metrics(
                manifest_records.get(domain, []),
                manifest_path=video_manifest,
                domain=domain,
                split=str(manifest_split),
            )
            last_metrics = current
            result.update(
                {f"{domain}_{name}": value for name, value in current.items()}
            )
            end_to_end_values.append(
                float(current["end_to_end_video_mean_precision"])
            )
        if len(domains) == 1:
            result.update(last_metrics)
        result["macro_end_to_end_video_mean_precision"] = sum(
            end_to_end_values
        ) / len(end_to_end_values)
    return result


def _selection_metric_name(
    *, multidomain: bool, split_manifest: Path | None
) -> str:
    if split_manifest is not None:
        return (
            "macro_end_to_end_video_mean_precision"
            if multidomain
            else "end_to_end_video_mean_precision"
        )
    return "macro_video_mean_precision" if multidomain else "video_mean_precision"


def _model_version(
    args: argparse.Namespace, *, multidomain: bool, person_mil: bool
) -> str:
    if args.fusion_mode == FYGR_FUSION_TEMPORAL_TOKENS:
        if person_mil:
            return (
                "v3.8-md-person-mil-temporal-token"
                if multidomain
                else "v3.8-person-mil-temporal-token"
            )
        return (
            "v3.8-md-temporal-token"
            if multidomain
            else "v3.8-temporal-token"
        )
    if person_mil:
        return "v3.7-md-person-mil" if multidomain else "v3.7-person-mil"
    if args.fusion_mode == FYGR_FUSION_CROSS_ATTENTION:
        if float(
            getattr(args, "fusion_hard_tail_ranking_loss_weight", 0.0)
        ) > 0:
            return (
                "v3.10-md-hard-tail-ranking"
                if multidomain
                else "v3.10-hard-tail-ranking"
            )
        if float(
            getattr(args, "fusion_hard_adl_ranking_loss_weight", 0.0)
        ) > 0:
            return (
                "v3.9-md-summary-ranking"
                if multidomain
                else "v3.9-summary-ranking"
            )
        return "v3.6-md-gcva" if multidomain else "v3.6-gcva"
    return "v3.5-md-fygr" if multidomain else "v3.4-fygr"


def _checkpoint_payload(
    args: argparse.Namespace,
    model: FrozenYoloGlobalROIConfirmer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_metric_name: str,
    best_metric: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_manifest: dict[str, Any],
) -> dict[str, Any]:
    multidomain = bool(args.extra_train_cache or args.extra_val_cache)
    person_mil = cache_person_capacity(train_manifest) > 1
    version = _model_version(
        args, multidomain=multidomain, person_mil=person_mil
    )
    return {
        "version": version,
        "epoch": epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "arguments": {
            key: _argument_value(value)
            for key, value in vars(args).items()
        },
        "feature_contract": train_manifest,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def run_training(args: argparse.Namespace) -> dict[str, float]:
    _validate(args)
    device = _device(args.device)
    _seed(args.seed)
    run_id, log_dir, checkpoint_dir, logger = _logger(args, device)
    try:
        selected_indices = (
            None
            if args.train_candidates is None
            else _candidate_indices(args.train_candidates)
        )
        selections = _selection_paths(args.split_manifest)
        train_datasets = [
            _cache_dataset(
                args.train_cache,
                split="train",
                selections=selections,
                shard_cache_size=args.shard_cache_size,
                selected_indices=selected_indices,
            )
        ]
        train_datasets.extend(
            _cache_dataset(
                root,
                split="train",
                selections=selections,
                shard_cache_size=args.shard_cache_size,
            )
            for root in args.extra_train_cache
        )
        train_datasets.extend(
            _cache_dataset(
                root,
                split="train",
                selections=selections,
                shard_cache_size=args.shard_cache_size,
            )
            for root in args.train_overlay_cache
        )
        val_datasets = [
            _cache_dataset(
                args.val_cache,
                split="val",
                selections=selections,
                shard_cache_size=args.shard_cache_size,
            )
        ]
        val_datasets.extend(
            _cache_dataset(
                root,
                split="val",
                selections=selections,
                shard_cache_size=args.shard_cache_size,
            )
            for root in args.extra_val_cache
        )
        train = _combine_datasets(train_datasets)
        val = _combine_datasets(val_datasets)
        _compatible_manifests(train.manifest, val.manifest)
        hard_negatives = (
            empty_hard_negative_registry()
            if args.hard_negative_manifest is None
            else load_hard_negative_registry(
                args.hard_negative_manifest,
                train.windows,
                positive_paths={
                    path
                    for path, value in train.video_has_fall_event.items()
                    if value
                },
            )
        )
        positive_paths = {
            path for path, value in train.video_has_fall_event.items() if value
        }
        hard_positives = (
            empty_hard_positive_registry()
            if args.hard_positive_manifest is None
            else load_hard_positive_registry(
                args.hard_positive_manifest,
                train.windows,
                positive_paths=positive_paths,
            )
        )
        global_shape = train.manifest["arrays"]["global_features"]
        roi_shape = roi_feature_shape(train.manifest)
        model = FrozenYoloGlobalROIConfirmer(
            geometry_checkpoint=args.geometry_checkpoint,
            history_offsets=train.history_offsets,
            visual_offsets=train.visual_offsets,
            global_feature_dim=int(global_shape[-1]),
            roi_feature_shape=roi_shape,
            temporal_dim=args.temporal_dim,
            dropout=args.dropout,
            use_global=args.use_global,
            use_roi=args.use_roi,
            freeze_geometry=not args.train_geometry,
            fusion_mode=args.fusion_mode,
            attention_heads=args.attention_heads,
            attention_ffn_dim=args.attention_ffn_dim,
            person_mil_temperature=args.person_mil_temperature,
        ).to(device)
        criterion = _criterion(args, hard_negatives).to(device)
        optimizer = _optimizer(args, model)
        scheduler = _scheduler(args, optimizer)
        amp_enabled = args.amp and device.type == "cuda"
        amp_dtype = (
            torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        )
        if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 is unsupported; falling back to float16 AMP")
            amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and amp_dtype == torch.float16
        )
        _update_parameters(
            log_dir,
            feature_contract=train.manifest,
            model=model,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            hard_negative_summary=dict(hard_negatives.summary),
            hard_positive_summary=dict(hard_positives.summary),
        )
        train_loader, val_loader, sampler = _make_loaders(
            args, train, val, hard_negatives, hard_positives
        )
        multidomain = bool(args.extra_train_cache or args.extra_val_cache)
        selection_metric_name = _selection_metric_name(
            multidomain=multidomain,
            split_manifest=args.split_manifest,
        )
        start_epoch = 1
        best_metric = float("-inf")
        if args.resume is not None:
            checkpoint = torch.load(
                args.resume, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            saved_metric_name = checkpoint.get("best_metric_name")
            if saved_metric_name != selection_metric_name:
                raise ValueError(
                    "Resume checkpoint model-selection metric differs from "
                    f"the current run: {saved_metric_name!r} != "
                    f"{selection_metric_name!r}"
                )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint["best_metric_value"])
        logger.info("run_id=%s checkpoints=%s", run_id, checkpoint_dir)
        logger.info(
            "data train=%d val=%d train_batches=%d val_batches=%d",
            len(train), len(val), len(train_loader), len(val_loader)
        )
        if hard_negatives.enabled:
            logger.info(
                "hard_negatives candidates=%d semantic=%d score=%d "
                "hard_windows=%d semantic_slots=%d score_slots=%d "
                "hard_ranking_weight=%.4f categories=%s domains=%s",
                hard_negatives.summary["candidate_count"],
                hard_negatives.summary["semantic_candidate_count"],
                hard_negatives.summary["score_candidate_count"],
                hard_negatives.summary["hard_window_count"],
                args.semantic_hard_videos_per_batch,
                args.score_hard_videos_per_batch,
                args.hard_video_ranking_loss_weight,
                hard_negatives.summary["category_counts"],
                hard_negatives.summary["domain_counts"],
            )
            logger.info(
                "fusion_hard_adl_ranking weight=%.4f margin=%.4f "
                "categories=%s anchor_count=%d",
                args.fusion_hard_adl_ranking_loss_weight,
                args.fusion_hard_adl_ranking_margin,
                _fusion_hard_adl_categories(args),
                len(hard_negatives.anchor_window_categories),
            )
            logger.info(
                "fusion_hard_tail_ranking weight=%.4f margin=%.4f "
                "negative_topk=%d positive_quantile=%.3f categories=%s "
                "chunk_window_count=%d",
                args.fusion_hard_tail_ranking_loss_weight,
                args.fusion_hard_tail_ranking_margin,
                args.fusion_hard_tail_negative_topk,
                args.fusion_hard_tail_positive_quantile,
                _fusion_hard_tail_categories(args),
                len(hard_negatives.window_categories),
            )
        if hard_positives.enabled:
            logger.info(
                "hard_positives candidates=%d videos=%d slots=%d domains=%s",
                hard_positives.summary["candidate_count"],
                hard_positives.summary["video_count"],
                args.hard_positive_videos_per_batch,
                hard_positives.summary["domain_counts"],
            )
        person_mil = cache_person_capacity(train.manifest) > 1
        model_version = _model_version(
            args, multidomain=multidomain, person_mil=person_mil
        )
        logger.info(
            "model version=%s cached_model_parameters=%d "
            "trainable_parameters=%d yolo_parameters_excluded=true "
            "use_global=%s use_roi=%s fusion_mode=%s geometry_frozen=%s "
            "geometry_lr=%.8f actor_top_k=%d person_mil_temperature=%.3f "
            "amp=%s dtype=%s",
            model_version
            + ("-joint-geometry" if args.train_geometry else ""),
            model.parameter_count,
            model.trainable_parameter_count,
            args.use_global,
            args.use_roi,
            args.fusion_mode,
            model.freeze_geometry,
            (
                args.learning_rate * args.geometry_lr_factor
                if args.train_geometry
                else 0.0
            ),
            cache_person_capacity(train.manifest),
            args.person_mil_temperature,
            amp_enabled,
            amp_dtype,
        )
        logger.info("model_selection_metric=%s", selection_metric_name)
        last_val: dict[str, float] = {}
        for epoch in range(start_epoch, args.epochs + 1):
            sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model,
                criterion,
                train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                gradient_clip=args.gradient_clip,
                history_truncation_prob=args.history_truncation_prob,
                max_batches=args.max_train_batches,
                log_interval=args.log_interval,
                logger=logger,
                epoch=epoch,
                phase="train",
            )
            val_metrics = run_epoch(
                model,
                criterion,
                val_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                gradient_clip=0.0,
                history_truncation_prob=0.0,
                max_batches=args.max_val_batches,
                log_interval=args.log_interval,
                logger=logger,
                epoch=epoch,
                phase="val",
                video_manifest=args.split_manifest,
                manifest_split=(
                    "val" if args.split_manifest is not None else None
                ),
            )
            last_val = val_metrics
            scheduler.step()
            metric_name = selection_metric_name
            metric = float(val_metrics.get(metric_name, 0.0))
            improved = metric > best_metric
            if improved:
                best_metric = metric
            payload = _checkpoint_payload(
                args,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_metric_name=metric_name,
                best_metric=best_metric,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                train_manifest=train.manifest,
            )
            torch.save(payload, checkpoint_dir / "last.pt")
            torch.save(payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
            if improved:
                torch.save(payload, checkpoint_dir / "best.pt")
            logger.info(
                "epoch=%d train %s | val %s best_%s=%.4f",
                epoch,
                _format_metrics(train_metrics),
                _format_metrics(val_metrics),
                metric_name,
                best_metric,
            )
        with (checkpoint_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(last_val, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return last_val
    except BaseException:
        logger.exception("training_failed")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
