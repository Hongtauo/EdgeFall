"""Command-line training entry point for EdgeFall-ST."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from edgefall_st.data import (
    OmniFallWindowDataset,
    omnifall_collate_fn,
    omnifall_indexed_collate_fn,
)
from edgefall_st.models import BBoxFallClassifier, EdgeFallST
from edgefall_st.models.bbox_classifier import (
    V32_LONG_HISTORY_OFFSETS,
    V32_SHORT_HISTORY_OFFSETS,
)
from edgefall_st.training.assignment import AnchorFreeAssigner
from edgefall_st.training.bbox_criterion import (
    BBoxClassifierLossWeights,
    BBoxFallCriterion,
)
from edgefall_st.training.criterion import EdgeFallCriterion, LossWeights
from edgefall_st.training.sampler import VideoGroupedBatchSampler


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    log_dir: Path
    checkpoint_dir: Path
    log_path: Path
    parameters_path: Path
    logger: logging.Logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train EdgeFall-ST on OmniFall windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    data.add_argument(
        "--train-windows",
        type=Path,
        default=Path("artifacts/omnifall_v1/windows_train.jsonl"),
    )
    data.add_argument(
        "--val-windows",
        type=Path,
        default=Path("artifacts/omnifall_v1/windows_val.jsonl"),
    )
    data.add_argument(
        "--train-tracks-root",
        type=Path,
        default=Path("artifacts/pseudo_labels/train_yolo26x_nms/tracks"),
    )
    data.add_argument(
        "--val-tracks-root",
        type=Path,
        default=Path("artifacts/pseudo_labels/val_yolo26x_nms/tracks"),
    )
    data.add_argument(
        "--train-actor-targets-root",
        type=Path,
        default=Path("artifacts/actor_targets_v1/train/targets"),
    )
    data.add_argument(
        "--val-actor-targets-root",
        type=Path,
        default=Path("artifacts/actor_targets_v1/val/targets"),
    )
    data.add_argument("--train-actor-crops-root", type=Path)
    data.add_argument("--val-actor-crops-root", type=Path)
    data.add_argument("--effective-fps", type=float, default=10.0)
    data.add_argument("--video-suffix", default=".mp4")
    data.add_argument("--onset-duration", type=float, default=0.5)
    data.add_argument("--annotation-cache-size", type=_positive_int, default=32)
    data.add_argument(
        "--video-frame-cache-size",
        type=_nonnegative_int,
        default=4,
        help="Number of resized uint8 videos cached per DataLoader worker",
    )
    data.add_argument(
        "--uint8-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep decoded images as uint8 until they reach the training device",
    )
    data.add_argument(
        "--deduplicate-backbone-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run shared source frames through the Backbone once per batch",
    )
    data.add_argument(
        "--validate-files",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--training-mode",
        choices=("unified", "bbox-classifier"),
        default="unified",
        help=(
            "unified learns person localization and semantics; "
            "bbox-classifier consumes existing actor boxes and only learns "
            "fall semantics"
        ),
    )
    model.add_argument("--input-height", type=_positive_int, default=288)
    model.add_argument("--input-width", type=_positive_int, default=512)
    model.add_argument(
        "--backbone-channels",
        type=_positive_int,
        nargs=5,
        default=(16, 32, 64, 128, 192),
        metavar=("C1", "C2", "C3", "C4", "C5"),
    )
    model.add_argument("--neck-channels", type=_positive_int, default=96)
    model.add_argument("--embedding-dim", type=_positive_int, default=8)
    model.add_argument(
        "--crop-height",
        type=_positive_int,
        default=128,
        help="Actor crop height used by bbox-classifier",
    )
    model.add_argument(
        "--crop-width",
        type=_positive_int,
        default=128,
        help="Actor crop width used by bbox-classifier",
    )
    model.add_argument(
        "--crop-context",
        type=float,
        default=1.25,
        help="Context expansion around supplied actor boxes",
    )
    model.add_argument(
        "--classifier-visual-dim",
        type=_positive_int,
        default=128,
    )
    model.add_argument(
        "--classifier-temporal-dim",
        type=_positive_int,
        default=128,
    )
    model.add_argument(
        "--classifier-temporal-model",
        choices=("attention", "dual-tcn", "v32", "v33-g", "v33-gr"),
        default="dual-tcn",
        help="Temporal fusion used by bbox-classifier",
    )
    model.add_argument(
        "--classifier-spatial-pool-size",
        type=_positive_int,
        help=(
            "Spatial pooling grid per backbone scale; defaults to 2 for "
            "V3.2/V3.3-GR and 1 for geometry-only/legacy classifiers"
        ),
    )
    model.add_argument(
        "--classifier-temporal-dropout",
        type=_unit_float,
        default=0.1,
    )
    model.add_argument(
        "--classifier-visual-modality-dropout",
        type=_unit_float,
        default=0.3,
        help="Per-sample RGB residual dropout used by V3.3-GR",
    )
    model.add_argument(
        "--classifier-visual-gate-bias",
        type=float,
        default=-2.0,
        help="Initial V3.3-GR geometry-uncertainty gate bias",
    )
    model.add_argument(
        "--classifier-visual-input-size",
        type=_positive_int,
        default=64,
        help=(
            "Square RGB confirmation input used by V3.3-GR; source actor "
            "crops remain at their configured storage size"
        ),
    )
    model.add_argument(
        "--classifier-geometry-init-checkpoint",
        type=Path,
        help="V3.3-G checkpoint used to initialize V3.3-GR geometry",
    )
    model.add_argument(
        "--classifier-freeze-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze initialized V3.3-G geometry while training RGB residuals",
    )
    model.add_argument(
        "--classifier-short-horizon",
        type=_positive_int,
        default=4,
        help="Maximum history offset used by the short-term TCN branch",
    )
    model.add_argument(
        "--classifier-v32-short-offsets",
        type=_nonnegative_int,
        nargs="+",
        default=V32_SHORT_HISTORY_OFFSETS,
        help="Regular recent-frame offsets used by V3.2/V3.3 short TCNs",
    )
    model.add_argument(
        "--classifier-v32-long-offsets",
        type=_nonnegative_int,
        nargs="+",
        default=V32_LONG_HISTORY_OFFSETS,
        help="Regular sparse offsets used by V3.2/V3.3 long TCNs",
    )
    model.add_argument(
        "--history-p3",
        type=_nonnegative_int,
        nargs="+",
        default=(0, 1, 2, 4),
    )
    model.add_argument(
        "--history-p4",
        type=_nonnegative_int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 24, 31),
    )
    model.add_argument(
        "--history-p5",
        type=_nonnegative_int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 24, 31),
    )
    model.add_argument(
        "--activation",
        choices=("silu", "relu"),
        default="silu",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=_positive_int, default=50)
    training.add_argument("--batch-size", type=_positive_int, default=8)
    training.add_argument(
        "--windows-per-video",
        type=_positive_int,
        default=2,
    )
    training.add_argument(
        "--positive-videos-per-batch",
        type=_nonnegative_int,
        default=1,
        help=(
            "Event videos per bbox-classifier batch; 0 disables "
            "event-balanced sampling"
        ),
    )
    training.add_argument(
        "--classifier-event-stage-balanced",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Rotate event chunks across falling/fallen phases; defaults to "
            "enabled for V3.3 and disabled for earlier classifiers"
        ),
    )
    training.add_argument(
        "--history-truncation-prob",
        type=_unit_float,
        default=0.2,
        help=(
            "Probability of randomly shortening bbox-classifier history "
            "during training"
        ),
    )
    training.add_argument(
        "--horizontal-flip-prob",
        type=_unit_float,
        default=0.5,
        help="Temporally consistent horizontal flip probability",
    )
    training.add_argument(
        "--grayscale-prob",
        type=_unit_float,
        default=0.2,
        help="Temporally consistent grayscale probability",
    )
    training.add_argument(
        "--photometric-jitter",
        type=_unit_float,
        default=0.15,
        help="Shared brightness and contrast jitter magnitude",
    )
    training.add_argument(
        "--bbox-jitter",
        type=_unit_float,
        default=0.05,
        help="Shared relative translation and scale jitter for actor boxes",
    )
    training.add_argument("--workers", type=_nonnegative_int, default=4)
    training.add_argument("--prefetch-factor", type=_positive_int, default=2)
    training.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    training.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    training.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--backbone-lr-factor", type=float, default=0.25)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--warmup-epochs", type=_nonnegative_int, default=3)
    training.add_argument("--minimum-lr-ratio", type=_unit_float, default=0.05)
    training.add_argument("--freeze-backbone-epochs", type=_nonnegative_int, default=2)
    training.add_argument("--gradient-clip", type=float, default=10.0)
    training.add_argument(
        "--accumulation-steps",
        type=_positive_int,
        default=1,
    )
    training.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    training.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
    )
    training.add_argument(
        "--initial-loss-scale",
        type=float,
        default=128.0,
    )
    training.add_argument("--seed", type=int, default=20260719)
    training.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    training.add_argument("--device", default="auto")
    training.add_argument("--resume", type=Path)
    training.add_argument("--log-interval", type=_positive_int, default=20)
    training.add_argument("--validate-every", type=_positive_int, default=1)
    training.add_argument("--save-every", type=_positive_int, default=5)
    training.add_argument(
        "--max-train-batches",
        type=_nonnegative_int,
        default=0,
        help="0 uses the complete training loader",
    )
    training.add_argument(
        "--max-val-batches",
        type=_nonnegative_int,
        default=0,
        help="0 uses the complete validation loader",
    )

    assignment = parser.add_argument_group("assignment")
    assignment.add_argument(
        "--regression-ranges",
        type=float,
        nargs=6,
        default=(0.0, 64.0, 64.0, 128.0, 128.0, math.inf),
        metavar=("P3_MIN", "P3_MAX", "P4_MIN", "P4_MAX", "P5_MIN", "P5_MAX"),
    )
    assignment.add_argument("--center-sampling-radius", type=float, default=1.5)
    assignment.add_argument(
        "--min-box-confidence",
        type=_unit_float,
        default=0.0,
    )

    losses = parser.add_argument_group("loss")
    losses.add_argument("--box-loss-weight", type=float, default=5.0)
    losses.add_argument("--objectness-loss-weight", type=float, default=1.0)
    losses.add_argument("--stage-loss-weight", type=float, default=2.0)
    losses.add_argument("--onset-loss-weight", type=float, default=1.0)
    losses.add_argument("--action-loss-weight", type=float, default=0.5)
    losses.add_argument("--track-loss-weight", type=float, default=0.1)
    losses.add_argument("--objectness-alpha", type=_unit_float, default=0.25)
    losses.add_argument("--objectness-gamma", type=float, default=2.0)
    losses.add_argument("--stage-gamma", type=float, default=2.0)
    losses.add_argument("--onset-alpha", type=_unit_float, default=0.75)
    losses.add_argument("--onset-gamma", type=float, default=2.0)
    losses.add_argument(
        "--stage-class-weights",
        type=float,
        nargs=3,
        default=(1.0, 2.0, 2.0),
    )
    losses.add_argument(
        "--action-class-weights",
        type=float,
        nargs=16,
        default=(1.0,) * 16,
    )
    losses.add_argument("--track-temperature", type=float, default=0.1)
    losses.add_argument(
        "--classifier-stage-loss-weight",
        type=float,
        default=1.0,
    )
    losses.add_argument(
        "--classifier-fall-loss-weight",
        type=float,
        default=2.0,
    )
    losses.add_argument(
        "--classifier-event-state-loss-weight",
        type=float,
        help=(
            "Separate falling/fallen event-state loss; defaults to 0.5 for "
            "V3.3 and 0 for earlier classifiers"
        ),
    )
    losses.add_argument(
        "--classifier-onset-loss-weight",
        type=float,
        default=0.5,
    )
    losses.add_argument(
        "--classifier-action-loss-weight",
        type=float,
        default=0.05,
        help="Low-weight 16-class auxiliary supervision for hard negatives",
    )
    losses.add_argument(
        "--classifier-control-loss-weight",
        type=float,
        help=(
            "Controlled-descent auxiliary loss; defaults to 0.25 for "
            "V3.2/V3.3 and 0 for legacy classifiers"
        ),
    )
    losses.add_argument(
        "--classifier-geometry-stage-loss-weight",
        type=float,
        help="Independent geometry stage loss; defaults to 0.5 for V3.3",
    )
    losses.add_argument(
        "--classifier-geometry-fall-loss-weight",
        type=float,
        help="Independent geometry event loss; defaults to 1.0 for V3.3",
    )
    losses.add_argument(
        "--classifier-consistency-loss-weight",
        type=float,
        default=0.25,
        help="Agreement penalty between binary fall and stage probabilities",
    )
    losses.add_argument(
        "--classifier-video-mil-loss-weight",
        type=float,
        default=0.5,
        help="Video-chunk pooled focal loss weight",
    )
    losses.add_argument(
        "--classifier-video-pooling",
        choices=("max", "topk-mean", "asymmetric"),
        help=(
            "Video-chunk pooling used for MIL/ranking; V3.3 defaults to "
            "event mean for positives and max for negatives"
        ),
    )
    losses.add_argument(
        "--classifier-video-topk",
        type=_positive_int,
        default=2,
        help="Number of windows averaged by topk-mean video pooling",
    )
    losses.add_argument(
        "--classifier-video-ranking-loss-weight",
        type=float,
        default=0.25,
        help="Positive/negative video-chunk ranking loss weight",
    )
    losses.add_argument(
        "--classifier-video-ranking-margin",
        type=float,
        default=0.2,
    )
    losses.add_argument(
        "--fall-alpha",
        type=_unit_float,
        default=0.75,
    )
    losses.add_argument("--fall-gamma", type=float, default=2.0)

    output = parser.add_argument_group("output")
    output.add_argument("--log-root", type=Path, default=Path("log"))
    output.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    return parser


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    return value


def _create_run_artifacts(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> RunArtifacts:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = args.log_root / run_id
    checkpoint_dir = args.checkpoint_root / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_path = log_dir / "train.log"
    parameters_path = log_dir / "parameters.yaml"

    logger = logging.getLogger(f"edgefall_st.train.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    payload = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": _serializable(vars(args)),
        "resolved": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": str(torch.__version__),
            "opencv_version": str(cv2.__version__),
            "numpy_version": str(np.__version__),
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }
    if args.training_mode == "bbox-classifier":
        payload["resolved"]["bbox_classifier"] = {
            "history_offsets": list(_classifier_history_offsets(args)),
            "spatial_pool_size": _classifier_spatial_pool_size(args),
            "control_loss_weight": _classifier_control_loss_weight(args),
            "event_state_loss_weight": _classifier_versioned_loss_weight(
                args,
                "classifier_event_state_loss_weight",
                v33_default=0.5,
            ),
            "geometry_stage_loss_weight": _classifier_versioned_loss_weight(
                args,
                "classifier_geometry_stage_loss_weight",
                v33_default=0.5,
                v33_g_default=0.0,
            ),
            "geometry_fall_loss_weight": _classifier_versioned_loss_weight(
                args,
                "classifier_geometry_fall_loss_weight",
                v33_default=1.0,
                v33_g_default=0.0,
            ),
            "event_stage_balanced": _classifier_event_stage_balanced(args),
            "video_pooling": _classifier_video_pooling(args),
            "video_topk": int(getattr(args, "classifier_video_topk", 2)),
        }
    with parameters_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    return RunArtifacts(
        run_id=run_id,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        log_path=log_path,
        parameters_path=parameters_path,
        logger=logger,
    )


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size % args.windows_per_video != 0:
        raise ValueError("batch-size must be divisible by windows-per-video")
    crop_roots = (
        args.train_actor_crops_root,
        args.val_actor_crops_root,
    )
    if (crop_roots[0] is None) != (crop_roots[1] is None):
        raise ValueError(
            "train/val actor crop roots must be configured together"
        )
    if args.training_mode != "bbox-classifier" and crop_roots[0] is not None:
        raise ValueError("Actor crop caches require bbox-classifier mode")
    videos_per_batch = args.batch_size // args.windows_per_video
    if (
        args.training_mode == "bbox-classifier"
        and not 0
        <= args.positive_videos_per_batch
        < videos_per_batch
    ):
        raise ValueError(
            "positive-videos-per-batch must be in [0, videos-per-batch)"
        )
    if (
        args.training_mode == "bbox-classifier"
        and args.positive_videos_per_batch
        and not args.drop_last
    ):
        raise ValueError(
            "event-balanced sampling requires --drop-last"
        )
    if args.effective_fps <= 0 or args.onset_duration < 0:
        raise ValueError("Invalid temporal data parameters")
    if args.learning_rate <= 0 or args.backbone_lr_factor <= 0:
        raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0 or args.gradient_clip < 0:
        raise ValueError("weight-decay and gradient-clip cannot be negative")
    if args.initial_loss_scale <= 0:
        raise ValueError("initial-loss-scale must be positive")
    if args.crop_context < 1.0:
        raise ValueError("crop-context must be at least 1")
    if not 0.0 <= args.classifier_temporal_dropout < 1.0:
        raise ValueError("classifier-temporal-dropout must be in [0, 1)")
    if not 0.0 <= args.classifier_visual_modality_dropout <= 1.0:
        raise ValueError(
            "classifier-visual-modality-dropout must be in [0, 1]"
        )
    geometry_init = args.classifier_geometry_init_checkpoint
    if geometry_init is not None:
        if (
            args.training_mode != "bbox-classifier"
            or args.classifier_temporal_model != "v33-gr"
        ):
            raise ValueError(
                "classifier-geometry-init-checkpoint requires v33-gr"
            )
        if args.resume is not None:
            raise ValueError(
                "classifier-geometry-init-checkpoint cannot be combined "
                "with resume"
            )
    if args.classifier_freeze_geometry and geometry_init is None:
        raise ValueError(
            "classifier-freeze-geometry requires a geometry checkpoint"
        )
    for name in ("history_p3", "history_p4", "history_p5"):
        offsets = tuple(getattr(args, name))
        if not offsets or offsets[0] != 0:
            raise ValueError(f"{name} must start with 0")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError(f"{name} must be unique and increasing")
    for name in (
        "classifier_v32_short_offsets",
        "classifier_v32_long_offsets",
    ):
        offsets = tuple(getattr(args, name))
        if not offsets or offsets[0] != 0:
            raise ValueError(f"{name} must start with 0")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError(f"{name} must be unique and increasing")
    if any(value < 0 for value in args.regression_ranges):
        raise ValueError("regression-ranges cannot contain negative values")
    ranges = tuple(args.regression_ranges)
    for index in range(0, len(ranges), 2):
        if ranges[index + 1] <= ranges[index]:
            raise ValueError("Each regression range must have min < max")
    loss_values = (
        args.box_loss_weight,
        args.objectness_loss_weight,
        args.stage_loss_weight,
        args.onset_loss_weight,
        args.action_loss_weight,
        args.track_loss_weight,
    )
    if any(value < 0 for value in loss_values):
        raise ValueError("Loss weights cannot be negative")
    classifier_loss_values = (
        args.classifier_stage_loss_weight,
        args.classifier_fall_loss_weight,
        _classifier_versioned_loss_weight(
            args,
            "classifier_event_state_loss_weight",
            v33_default=0.5,
        ),
        args.classifier_onset_loss_weight,
        args.classifier_action_loss_weight,
        _classifier_control_loss_weight(args),
        _classifier_versioned_loss_weight(
            args,
            "classifier_geometry_stage_loss_weight",
            v33_default=0.5,
            v33_g_default=0.0,
        ),
        _classifier_versioned_loss_weight(
            args,
            "classifier_geometry_fall_loss_weight",
            v33_default=1.0,
            v33_g_default=0.0,
        ),
        args.classifier_consistency_loss_weight,
        args.classifier_video_mil_loss_weight,
        args.classifier_video_ranking_loss_weight,
    )
    if any(value < 0 for value in classifier_loss_values):
        raise ValueError("Classifier loss weights cannot be negative")
    if args.fall_gamma < 0:
        raise ValueError("fall-gamma cannot be negative")
    if args.classifier_video_ranking_margin < 0:
        raise ValueError("classifier-video-ranking-margin cannot be negative")


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _history_offsets(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    return {
        "p3": tuple(args.history_p3),
        "p4": tuple(args.history_p4),
        "p5": tuple(args.history_p5),
    }


def _global_history_offsets(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                offset
                for offsets in _history_offsets(args).values()
                for offset in offsets
            }
        )
    )


def _classifier_history_offsets(args: argparse.Namespace) -> tuple[int, ...]:
    temporal_model = getattr(args, "classifier_temporal_model", "attention")
    if temporal_model not in {"v32", "v33-g", "v33-gr"}:
        return _global_history_offsets(args)
    short_offsets = tuple(
        int(value)
        for value in getattr(
            args,
            "classifier_v32_short_offsets",
            V32_SHORT_HISTORY_OFFSETS,
        )
    )
    long_offsets = tuple(
        int(value)
        for value in getattr(
            args,
            "classifier_v32_long_offsets",
            V32_LONG_HISTORY_OFFSETS,
        )
    )
    return tuple(sorted(set(short_offsets).union(long_offsets)))


def _classifier_spatial_pool_size(args: argparse.Namespace) -> int:
    configured = getattr(args, "classifier_spatial_pool_size", None)
    if configured is not None:
        return int(configured)
    return (
        2
        if getattr(args, "classifier_temporal_model", "attention")
        in {"v32", "v33-gr"}
        else 1
    )


def _classifier_control_loss_weight(args: argparse.Namespace) -> float:
    configured = getattr(args, "classifier_control_loss_weight", None)
    if configured is not None:
        return float(configured)
    return (
        0.25
        if getattr(args, "classifier_temporal_model", "attention")
        in {"v32", "v33-g", "v33-gr"}
        else 0.0
    )


def _classifier_versioned_loss_weight(
    args: argparse.Namespace,
    name: str,
    *,
    v33_default: float,
    v33_g_default: float | None = None,
) -> float:
    configured = getattr(args, name, None)
    if configured is not None:
        return float(configured)
    temporal_model = getattr(args, "classifier_temporal_model", "attention")
    if temporal_model == "v33-g":
        return float(
            v33_default if v33_g_default is None else v33_g_default
        )
    return float(v33_default) if temporal_model == "v33-gr" else 0.0


def _classifier_event_stage_balanced(args: argparse.Namespace) -> bool:
    configured = getattr(args, "classifier_event_stage_balanced", None)
    if configured is not None:
        return bool(configured)
    return getattr(args, "classifier_temporal_model", "attention") in {
        "v33-g",
        "v33-gr",
    }


def _classifier_video_pooling(args: argparse.Namespace) -> str:
    configured = getattr(args, "classifier_video_pooling", None)
    if configured is not None:
        return str(configured)
    temporal_model = getattr(args, "classifier_temporal_model", "attention")
    if temporal_model in {"v33-g", "v33-gr"}:
        return "asymmetric"
    return "topk-mean" if temporal_model == "v32" else "max"


def _make_datasets(
    args: argparse.Namespace,
) -> tuple[OmniFallWindowDataset, OmniFallWindowDataset]:
    input_size = (args.input_height, args.input_width)
    global_offsets = (
        _classifier_history_offsets(args)
        if args.training_mode == "bbox-classifier"
        else _global_history_offsets(args)
    )
    common = {
        "dataset_root": args.dataset_root,
        "input_size": input_size,
        "effective_fps": args.effective_fps,
        "history_offsets": global_offsets,
        "annotation_cache_size": args.annotation_cache_size,
        "video_frame_cache_size": args.video_frame_cache_size,
        "return_uint8": args.uint8_images,
        "onset_duration_s": args.onset_duration,
        "video_suffix": args.video_suffix,
        "actor_crop_size": (args.crop_height, args.crop_width),
        "validate_files": args.validate_files,
    }
    train_dataset = OmniFallWindowDataset(
        windows_path=args.train_windows,
        tracks_root=args.train_tracks_root,
        actor_targets_root=args.train_actor_targets_root,
        actor_crops_root=args.train_actor_crops_root,
        **common,
    )
    val_dataset = OmniFallWindowDataset(
        windows_path=args.val_windows,
        tracks_root=args.val_tracks_root,
        actor_targets_root=args.val_actor_targets_root,
        actor_crops_root=args.val_actor_crops_root,
        **common,
    )
    return train_dataset, val_dataset


def _loader_worker_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers == 0:
        return {}
    return {
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
    }


def _make_loaders(
    args: argparse.Namespace,
    train_dataset: OmniFallWindowDataset,
    val_dataset: OmniFallWindowDataset,
) -> tuple[DataLoader[Any], DataLoader[Any], VideoGroupedBatchSampler]:
    sampler = VideoGroupedBatchSampler(
        train_dataset.windows,
        batch_size=args.batch_size,
        windows_per_video=args.windows_per_video,
        shuffle=True,
        drop_last=args.drop_last,
        seed=args.seed,
        positive_paths=(
            {
                path
                for path, has_event in (
                    train_dataset.video_has_fall_event.items()
                )
                if has_event
            }
            if args.training_mode == "bbox-classifier"
            else None
        ),
        positive_videos_per_batch=(
            args.positive_videos_per_batch
            if args.training_mode == "bbox-classifier"
            else 0
        ),
        event_stage_balanced=(
            _classifier_event_stage_balanced(args)
            if args.training_mode == "bbox-classifier"
            else False
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    worker_options = _loader_worker_options(args)
    collate_fn = (
        omnifall_indexed_collate_fn
        if args.deduplicate_backbone_frames
        else omnifall_collate_fn
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        **worker_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        **worker_options,
    )
    return train_loader, val_loader, sampler


def _make_model_and_criterion(
    args: argparse.Namespace,
) -> tuple[
    EdgeFallST | BBoxFallClassifier,
    EdgeFallCriterion | BBoxFallCriterion,
]:
    input_size = (args.input_height, args.input_width)
    if args.training_mode == "bbox-classifier":
        model = BBoxFallClassifier(
            input_size=input_size,
            crop_size=(args.crop_height, args.crop_width),
            crop_context=args.crop_context,
            backbone_channels=tuple(args.backbone_channels),
            visual_dim=args.classifier_visual_dim,
            temporal_dim=args.classifier_temporal_dim,
            history_offsets=_classifier_history_offsets(args),
            activation=args.activation,
            temporal_model=getattr(
                args,
                "classifier_temporal_model",
                "attention",
            ),
            temporal_dropout=getattr(
                args,
                "classifier_temporal_dropout",
                0.1,
            ),
            short_horizon=getattr(
                args,
                "classifier_short_horizon",
                4,
            ),
            spatial_pool_size=_classifier_spatial_pool_size(args),
            short_history_offsets=getattr(
                args,
                "classifier_v32_short_offsets",
                V32_SHORT_HISTORY_OFFSETS,
            ),
            long_history_offsets=getattr(
                args,
                "classifier_v32_long_offsets",
                V32_LONG_HISTORY_OFFSETS,
            ),
            effective_fps=args.effective_fps,
            pre_cropped_inputs=(
                getattr(args, "train_actor_crops_root", None) is not None
            ),
            visual_modality_dropout=getattr(
                args,
                "classifier_visual_modality_dropout",
                0.3,
            ),
            visual_gate_bias=getattr(
                args,
                "classifier_visual_gate_bias",
                -2.0,
            ),
            visual_input_size=getattr(
                args,
                "classifier_visual_input_size",
                64,
            ),
        )
        criterion = BBoxFallCriterion(
            loss_weights=BBoxClassifierLossWeights(
                stage=args.classifier_stage_loss_weight,
                fall=args.classifier_fall_loss_weight,
                event_state=_classifier_versioned_loss_weight(
                    args,
                    "classifier_event_state_loss_weight",
                    v33_default=0.5,
                ),
                onset=args.classifier_onset_loss_weight,
                action=args.classifier_action_loss_weight,
                control=_classifier_control_loss_weight(args),
                geometry_stage=_classifier_versioned_loss_weight(
                    args,
                    "classifier_geometry_stage_loss_weight",
                    v33_default=0.5,
                    v33_g_default=0.0,
                ),
                geometry_fall=_classifier_versioned_loss_weight(
                    args,
                    "classifier_geometry_fall_loss_weight",
                    v33_default=1.0,
                    v33_g_default=0.0,
                ),
                consistency=getattr(
                    args,
                    "classifier_consistency_loss_weight",
                    0.0,
                ),
                video_mil=getattr(
                    args,
                    "classifier_video_mil_loss_weight",
                    0.0,
                ),
                video_ranking=getattr(
                    args,
                    "classifier_video_ranking_loss_weight",
                    0.0,
                ),
            ),
            stage_class_weights=args.stage_class_weights,
            action_class_weights=args.action_class_weights,
            stage_gamma=args.stage_gamma,
            fall_alpha=args.fall_alpha,
            fall_gamma=args.fall_gamma,
            onset_alpha=args.onset_alpha,
            onset_gamma=args.onset_gamma,
            video_ranking_margin=getattr(
                args,
                "classifier_video_ranking_margin",
                0.2,
            ),
            video_pooling=_classifier_video_pooling(args),
            video_topk=getattr(args, "classifier_video_topk", 2),
        )
        return model, criterion

    model = EdgeFallST(
        input_size=input_size,
        backbone_channels=tuple(args.backbone_channels),
        neck_channels=args.neck_channels,
        embedding_dim=args.embedding_dim,
        history_offsets=_history_offsets(args),
        activation=args.activation,
    )
    values = tuple(args.regression_ranges)
    regression_ranges = tuple(
        (values[index], values[index + 1])
        for index in range(0, len(values), 2)
    )
    assigner = AnchorFreeAssigner(
        input_size=input_size,
        regression_ranges=regression_ranges,
        center_sampling_radius=args.center_sampling_radius,
        min_box_confidence=args.min_box_confidence,
    )
    criterion = EdgeFallCriterion(
        input_size=input_size,
        assigner=assigner,
        loss_weights=LossWeights(
            box=args.box_loss_weight,
            objectness=args.objectness_loss_weight,
            stage=args.stage_loss_weight,
            onset=args.onset_loss_weight,
            action=args.action_loss_weight,
            track=args.track_loss_weight,
        ),
        objectness_alpha=args.objectness_alpha,
        objectness_gamma=args.objectness_gamma,
        stage_gamma=args.stage_gamma,
        onset_alpha=args.onset_alpha,
        onset_gamma=args.onset_gamma,
        stage_class_weights=args.stage_class_weights,
        action_class_weights=args.action_class_weights,
        track_temperature=args.track_temperature,
    )
    return model, criterion


def _make_optimizer(
    args: argparse.Namespace,
    model: EdgeFallST | BBoxFallClassifier,
) -> AdamW:
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    remaining_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    parameter_groups: list[dict[str, Any]] = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.learning_rate * args.backbone_lr_factor,
                "name": "backbone",
            }
        )
    if remaining_parameters:
        parameter_groups.append(
            {
                "params": remaining_parameters,
                "lr": args.learning_rate,
                "name": "main",
            }
        )
    if not parameter_groups:
        raise ValueError("Model has no trainable parameters")
    return AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )


def _make_scheduler(args: argparse.Namespace, optimizer: Optimizer) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        cosine_epochs = max(1, args.epochs - args.warmup_epochs)
        progress = min(
            1.0,
            max(0.0, (epoch - args.warmup_epochs) / cosine_epochs),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.minimum_lr_ratio + (
            1.0 - args.minimum_lr_ratio
        ) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _move_targets(
    targets: Sequence[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        {
            name: (
                value.to(device=device, non_blocking=True)
                if isinstance(value, Tensor)
                else value
            )
            for name, value in target.items()
        }
        for target in targets
    ]


class _MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_sums: dict[str, float] = {}
        self.sample_count = 0
        self.counts: dict[str, int] = {}

    def update(self, losses: dict[str, Tensor], batch_size: int) -> None:
        self.sample_count += batch_size
        for name, value in losses.items():
            scalar = float(value.detach().cpu())
            if name.startswith("num_"):
                self.counts[name] = self.counts.get(name, 0) + int(scalar)
            elif name.startswith("loss"):
                self.weighted_sums[name] = (
                    self.weighted_sums.get(name, 0.0)
                    + scalar * batch_size
                )

    def result(self) -> dict[str, float]:
        if self.sample_count == 0:
            raise RuntimeError("No batches were processed")
        result = {
            name: total / self.sample_count
            for name, total in self.weighted_sums.items()
        }
        result.update(
            {name: float(value) for name, value in self.counts.items()}
        )
        return result


def _binary_ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> dict[str, float]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if not scores:
        return {
            "average_precision": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "mean_precision": 0.0,
        }
    records = sorted(
        zip(scores, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    positive_count = sum(int(label) for label in labels)
    if positive_count == 0:
        return {
            "average_precision": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "mean_precision": 0.0,
        }

    curve: list[tuple[float, float]] = []
    true_positives = 0
    false_positives = 0
    average_precision = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(records):
        score = records[index][0]
        group_true_positives = 0
        group_size = 0
        while index < len(records) and records[index][0] == score:
            group_true_positives += int(records[index][1])
            group_size += 1
            index += 1
        true_positives += group_true_positives
        false_positives += group_size - group_true_positives
        recall = true_positives / positive_count
        precision = true_positives / max(
            true_positives + false_positives,
            1,
        )
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        curve.append((recall, precision))

    def precision_at_recall(target: float) -> float:
        return max(
            (
                precision
                for recall, precision in curve
                if recall >= target
            ),
            default=0.0,
        )

    p90 = precision_at_recall(0.90)
    p95 = precision_at_recall(0.95)
    return {
        "average_precision": average_precision,
        "p90": p90,
        "p95": p95,
        "mean_precision": (p90 + p95) * 0.5,
    }


class _ClassifierMetricAccumulator:
    def __init__(self) -> None:
        self.window_scores: list[float] = []
        self.window_labels: list[bool] = []
        self.video_scores: dict[str, float] = {}
        self.geometry_window_scores: list[float] = []
        self.geometry_video_scores: dict[str, float] = {}
        self.video_targets: dict[str, bool] = {}
        self.positive_videos: set[str] = set()
        self.stage_correct = 0
        self.supervised = 0
        self.fall_tp = 0
        self.fall_fp = 0
        self.fall_fn = 0
        self.onset_tp = 0
        self.onset_fp = 0
        self.onset_fn = 0
        self.visual_gate_sum = 0.0
        self.visual_gate_count = 0

    def update(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        metadata: Sequence[dict[str, Any]],
    ) -> None:
        fall_scores = outputs["fall_logits"].detach().float().sigmoid().cpu()
        onset_scores = outputs["onset_logits"].detach().float().sigmoid().cpu()
        stage_predictions = (
            outputs["stage_logits"].detach().float().argmax(dim=-1).cpu()
        )
        geometry_scores = None
        if "geometry_fall_logits" in outputs:
            geometry_scores = (
                outputs["geometry_fall_logits"]
                .detach()
                .float()
                .sigmoid()
                .cpu()
            )
        if "visual_gate" in outputs:
            gates = outputs["visual_gate"].detach().float().cpu()
            self.visual_gate_sum += float(gates.sum())
            self.visual_gate_count += int(gates.numel())
        for index, (target, sample_metadata) in enumerate(
            zip(targets, metadata, strict=True)
        ):
            stage_target = int(target["stage_label"].detach().cpu())
            onset_target = (
                float(target["onset_target"].detach().cpu()) >= 0.5
            )
            fall_target = bool(
                sample_metadata.get("event_target", stage_target != 0)
            )
            fall_score = float(fall_scores[index])
            onset_score = float(onset_scores[index])
            fall_prediction = fall_score >= 0.5
            onset_prediction = onset_score >= 0.5
            path = str(sample_metadata["path"])
            video_target = bool(
                sample_metadata.get("video_has_fall_event", fall_target)
            )
            previous_video_target = self.video_targets.get(path)
            if (
                previous_video_target is not None
                and previous_video_target != video_target
            ):
                raise ValueError(
                    f"Inconsistent video_has_fall_event for {path!r}"
                )
            self.video_targets[path] = video_target

            self.video_scores[path] = max(
                self.video_scores.get(path, 0.0),
                fall_score,
            )
            if geometry_scores is not None:
                geometry_score = float(geometry_scores[index])
                self.geometry_video_scores[path] = max(
                    self.geometry_video_scores.get(path, 0.0),
                    geometry_score,
                )
            if video_target:
                self.positive_videos.add(path)
            event_weight = target.get(
                "event_loss_weight", target["action_loss_weight"]
            )
            if float(event_weight.detach().cpu()) <= 0:
                continue

            self.supervised += 1
            self.stage_correct += int(
                int(stage_predictions[index]) == stage_target
            )
            self.window_scores.append(fall_score)
            self.window_labels.append(fall_target)
            if geometry_scores is not None:
                self.geometry_window_scores.append(float(geometry_scores[index]))
            if fall_prediction and fall_target:
                self.fall_tp += 1
            elif fall_prediction:
                self.fall_fp += 1
            elif fall_target:
                self.fall_fn += 1
            if onset_prediction and onset_target:
                self.onset_tp += 1
            elif onset_prediction:
                self.onset_fp += 1
            elif onset_target:
                self.onset_fn += 1

    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def result(self) -> dict[str, float]:
        if self.supervised == 0:
            return {}
        fall_precision = self._safe_divide(
            self.fall_tp,
            self.fall_tp + self.fall_fp,
        )
        fall_recall = self._safe_divide(
            self.fall_tp,
            self.fall_tp + self.fall_fn,
        )
        onset_precision = self._safe_divide(
            self.onset_tp,
            self.onset_tp + self.onset_fp,
        )
        onset_recall = self._safe_divide(
            self.onset_tp,
            self.onset_tp + self.onset_fn,
        )
        window_ranking = _binary_ranking_metrics(
            self.window_scores,
            self.window_labels,
        )
        video_paths = list(self.video_scores)
        video_ranking = _binary_ranking_metrics(
            [self.video_scores[path] for path in video_paths],
            [path in self.positive_videos for path in video_paths],
        )
        geometry_window_ranking = _binary_ranking_metrics(
            self.geometry_window_scores,
            self.window_labels if self.geometry_window_scores else (),
        )
        geometry_video_paths = list(self.geometry_video_scores)
        geometry_video_ranking = _binary_ranking_metrics(
            [self.geometry_video_scores[path] for path in geometry_video_paths],
            [path in self.positive_videos for path in geometry_video_paths],
        )
        return {
            "stage_accuracy": self.stage_correct / self.supervised,
            "fall_precision": fall_precision,
            "fall_recall": fall_recall,
            "fall_f1": (
                2.0 * fall_precision * fall_recall
                / max(fall_precision + fall_recall, 1e-12)
            ),
            "onset_precision": onset_precision,
            "onset_recall": onset_recall,
            **{
                f"window_{name}": value
                for name, value in window_ranking.items()
            },
            **{
                f"video_{name}": value
                for name, value in video_ranking.items()
            },
            **{
                f"geometry_window_{name}": value
                for name, value in geometry_window_ranking.items()
            },
            **{
                f"geometry_video_{name}": value
                for name, value in geometry_video_ranking.items()
            },
            "visual_gate_mean": (
                self.visual_gate_sum / self.visual_gate_count
                if self.visual_gate_count
                else 0.0
            ),
        }


def _format_metrics(metrics: dict[str, float]) -> str:
    ordered = (
        "loss",
        "loss_box",
        "loss_objectness",
        "loss_stage",
        "loss_fall",
        "loss_event_state",
        "loss_onset",
        "loss_action",
        "loss_control",
        "loss_geometry_stage",
        "loss_geometry_fall",
        "loss_consistency",
        "loss_video_mil",
        "loss_video_ranking",
        "loss_hard_video_ranking",
        "loss_global_aux",
        "loss_roi_aux",
        "loss_global_aux_video_mil",
        "loss_roi_aux_video_mil",
        "loss_fusion_hard_adl_ranking",
        "loss_fusion_hard_tail_ranking",
        "loss_track",
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "video_p90",
        "video_p95",
        "macro_video_mean_precision",
        "geometry_video_mean_precision",
        "visual_gate_mean",
        "fusion_state_norm_mean",
        "fusion_attention_entropy_mean",
        "fusion_global_mass_mean",
        "fusion_roi_mass_mean",
    )
    return " ".join(
        f"{name}={metrics[name]:.4f}"
        for name in ordered
        if name in metrics
    )


def _effective_batch_count(loader: DataLoader[Any], maximum: int) -> int:
    count = len(loader)
    return min(count, maximum) if maximum > 0 else count


def _truncate_history_mask(
    valid_mask: Tensor,
    probability: float,
) -> Tensor:
    """Randomly remove oldest slots while preserving the current frame."""

    if probability <= 0.0 or valid_mask.shape[1] <= 1:
        return valid_mask
    batch_size, slots = valid_mask.shape
    valid_counts = valid_mask.gt(0).sum(dim=1)
    selected = (
        torch.rand(batch_size, device=valid_mask.device) < probability
    ) & valid_counts.gt(1)
    choices = (valid_counts - 1).clamp_min(1)
    keep_counts = (
        torch.rand(batch_size, device=valid_mask.device) * choices
    ).floor().long() + 1
    slot_indices = torch.arange(slots, device=valid_mask.device)[None]
    truncated = valid_mask * slot_indices.lt(keep_counts[:, None]).to(
        valid_mask.dtype
    )
    return torch.where(selected[:, None], truncated, valid_mask)


def _augment_bbox_classifier_inputs(
    images: Tensor,
    actor_boxes: Tensor,
    actor_valid_mask: Tensor,
    *,
    horizontal_flip_prob: float,
    grayscale_prob: float,
    photometric_jitter: float,
    bbox_jitter: float,
    pre_cropped_inputs: bool = False,
    crop_context: float = 1.25,
    augment_images: bool = True,
) -> tuple[Tensor, Tensor]:
    """Apply one transform per batch so temporal motion stays coherent."""

    if images.shape[-3] != 3 or actor_boxes.shape[-1] != 4:
        raise ValueError("Unexpected bbox-classifier input shapes")
    if actor_boxes.shape[:-1] != actor_valid_mask.shape:
        raise ValueError("actor_boxes and actor_valid_mask shapes disagree")

    random_device = actor_boxes.device
    if torch.rand((), device=random_device) < horizontal_flip_prob:
        if augment_images:
            images = images.flip(-1)
        flipped_boxes = actor_boxes.clone()
        old_left = actor_boxes[..., 0]
        old_right = actor_boxes[..., 2]
        flipped_boxes[..., 0] = 1.0 - old_right
        flipped_boxes[..., 2] = 1.0 - old_left
        actor_boxes = torch.where(
            actor_valid_mask.gt(0)[..., None],
            flipped_boxes,
            actor_boxes,
        )

    if augment_images and photometric_jitter > 0.0:
        brightness = 1.0 + (
            2.0 * torch.rand((), device=images.device) - 1.0
        ) * photometric_jitter
        contrast = 1.0 + (
            2.0 * torch.rand((), device=images.device) - 1.0
        ) * photometric_jitter
        mean = images.mean(dim=(-2, -1), keepdim=True)
        images = ((images - mean) * contrast + mean) * brightness
        images = images.clamp(0.0, 1.0)

    if augment_images and torch.rand(
        (), device=random_device
    ) < grayscale_prob:
        coefficients = images.new_tensor((0.299, 0.587, 0.114))
        shape = [1] * images.ndim
        shape[-3] = 3
        grayscale = (
            images * coefficients.reshape(shape)
        ).sum(dim=-3, keepdim=True)
        images = grayscale.expand_as(images).contiguous()

    if bbox_jitter > 0.0:
        delta_x = (
            2.0 * torch.rand((), device=random_device) - 1.0
        ) * bbox_jitter
        delta_y = (
            2.0 * torch.rand((), device=random_device) - 1.0
        ) * bbox_jitter
        scale = 1.0 + (
            2.0 * torch.rand((), device=random_device) - 1.0
        ) * bbox_jitter
        if augment_images and pre_cropped_inputs:
            if crop_context < 1.0:
                raise ValueError("crop_context must be at least 1")
            flat_images = images.reshape(-1, 3, *images.shape[-2:])
            theta = images.new_zeros((flat_images.shape[0], 2, 3))
            theta[:, 0, 0] = scale
            theta[:, 1, 1] = scale
            theta[:, 0, 2] = 2.0 * delta_x / crop_context
            theta[:, 1, 2] = 2.0 * delta_y / crop_context
            grid = F.affine_grid(
                theta,
                flat_images.shape,
                align_corners=False,
            )
            images = F.grid_sample(
                flat_images,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).reshape_as(images)

        actor_boxes = actor_boxes.clone()
        left, top, right, bottom = actor_boxes.unbind(dim=-1)
        width = (right - left).clamp_min(1e-4)
        height = (bottom - top).clamp_min(1e-4)
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        center_x = center_x + delta_x * width
        center_y = center_y + delta_y * height
        width = width * scale
        height = height * scale
        jittered = torch.stack(
            (
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        valid = actor_valid_mask.gt(0)[..., None]
        actor_boxes = torch.where(valid, jittered, actor_boxes)
    return images, actor_boxes


def _run_epoch(
    *,
    model: EdgeFallST | BBoxFallClassifier,
    criterion: EdgeFallCriterion | BBoxFallCriterion,
    training_mode: str,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    accumulation_steps: int,
    gradient_clip: float,
    maximum_batches: int,
    logger: logging.Logger,
    log_interval: int,
    epoch: int,
    backbone_frozen: bool,
    history_truncation_prob: float,
    horizontal_flip_prob: float,
    grayscale_prob: float,
    photometric_jitter: float,
    bbox_jitter: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion.train(training)
    if training and backbone_frozen:
        model.backbone.eval()
    if training:
        optimizer.zero_grad(set_to_none=True)

    batch_count = _effective_batch_count(loader, maximum_batches)
    if batch_count == 0:
        raise RuntimeError("DataLoader contains no batches")
    meter = _MetricAccumulator()
    classifier_meter = (
        _ClassifierMetricAccumulator()
        if training_mode == "bbox-classifier"
        else None
    )
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    start_time = time.perf_counter()
    previous_batch_end = start_time
    data_time_s = 0.0
    compute_time_s = 0.0
    backbone_frames = 0
    valid_frame_slots = 0
    frame_slots = 0
    geometry_only_classifier = (
        isinstance(model, BBoxFallClassifier)
        and model.temporal_model == "v33-g"
    )
    gradient_context = torch.enable_grad if training else torch.no_grad
    with gradient_context():
        for batch_index, batch in enumerate(loader):
            if batch_index >= batch_count:
                break
            batch_ready = time.perf_counter()
            data_time_s += batch_ready - previous_batch_end
            compute_start = batch_ready
            source_images = batch["images"]
            if geometry_only_classifier:
                images = source_images
            elif source_images.dtype == torch.uint8:
                images = source_images.to(
                    device=device,
                    non_blocking=True,
                )
                images = images.to(dtype=torch.float32).div_(255.0)
            else:
                images = source_images.to(
                    device=device,
                    non_blocking=True,
                )
            valid_mask = batch["valid_mask"].to(
                device=device,
                non_blocking=True,
            )
            if training and training_mode == "bbox-classifier":
                valid_mask = _truncate_history_mask(
                    valid_mask,
                    history_truncation_prob,
                )
            history_indices = batch.get("history_indices")
            if history_indices is not None:
                history_indices = history_indices.to(
                    device=device,
                    non_blocking=True,
                )
            actor_boxes = batch["actor_boxes"].to(
                device=device,
                non_blocking=True,
            )
            actor_valid_mask = batch["actor_valid_mask"].to(
                device=device,
                non_blocking=True,
            )
            if training and training_mode == "bbox-classifier":
                images, actor_boxes = _augment_bbox_classifier_inputs(
                    images,
                    actor_boxes,
                    actor_valid_mask,
                    horizontal_flip_prob=horizontal_flip_prob,
                    grayscale_prob=grayscale_prob,
                    photometric_jitter=photometric_jitter,
                    bbox_jitter=bbox_jitter,
                    pre_cropped_inputs=model.pre_cropped_inputs,
                    crop_context=model.cropper.context_scale,
                    augment_images=not geometry_only_classifier,
                )
            targets = _move_targets(batch["targets"], device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if training_mode == "bbox-classifier":
                    if not isinstance(model, BBoxFallClassifier):
                        raise TypeError("bbox-classifier mode requires its model")
                    if history_indices is None:
                        outputs = model.forward_history(
                            images,
                            actor_boxes,
                            valid_mask,
                            actor_valid_mask,
                        )
                    else:
                        outputs = model.forward_indexed_history(
                            images,
                            actor_boxes,
                            history_indices,
                            valid_mask,
                            actor_valid_mask,
                        )
                else:
                    if not isinstance(model, EdgeFallST):
                        raise TypeError("unified mode requires EdgeFallST")
                    if history_indices is None:
                        outputs = model.forward_history(images, valid_mask)
                    else:
                        outputs = model.forward_indexed_history(
                            images,
                            history_indices,
                            valid_mask,
                        )
                losses = criterion(outputs, targets, batch["metadata"])
                scaled_loss = losses["loss"] / accumulation_steps

            if training:
                scaler.scale(scaled_loss).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == batch_count
                )
                if should_step:
                    if gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip,
                        )
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    if scaler.is_enabled() and scale_after < scale_before:
                        skipped_optimizer_steps += 1
                    else:
                        optimizer_steps += 1
                    optimizer.zero_grad(set_to_none=True)

            batch_size = len(batch["targets"])
            slots = int(valid_mask.shape[1])
            batch_backbone_frames = 0
            if not geometry_only_classifier:
                if isinstance(model, BBoxFallClassifier):
                    batch_backbone_frames = model.backbone_frame_count(
                        history_indices,
                        batch_size,
                    )
                else:
                    batch_backbone_frames = (
                        int(images.shape[0])
                        if history_indices is not None
                        else int(images.shape[0] * images.shape[1])
                    )
            backbone_frames += batch_backbone_frames
            valid_frame_slots += int(batch["valid_mask"].sum().item())
            frame_slots += batch_size * slots
            meter.update(losses, batch_size)
            if classifier_meter is not None:
                classifier_meter.update(
                    outputs,
                    targets,
                    batch["metadata"],
                )
            batch_end = time.perf_counter()
            compute_time_s += batch_end - compute_start
            previous_batch_end = batch_end
            if training and (
                (batch_index + 1) % log_interval == 0
                or batch_index + 1 == batch_count
            ):
                current = meter.result()
                elapsed = time.perf_counter() - start_time
                frame_reduction_pct = 100.0 * (
                    1.0 - backbone_frames / max(frame_slots, 1)
                )
                logger.info(
                    "epoch=%d train_batch=%d/%d %s samples_per_s=%.2f "
                    "data_time_s=%.1f compute_time_s=%.1f "
                    "backbone_frames=%d/%d valid_frame_slots=%d "
                    "frame_reduction_pct=%.1f",
                    epoch + 1,
                    batch_index + 1,
                    batch_count,
                    _format_metrics(current),
                    meter.sample_count / max(elapsed, 1e-6),
                    data_time_s,
                    compute_time_s,
                    backbone_frames,
                    frame_slots,
                    valid_frame_slots,
                    frame_reduction_pct,
                )
    metrics = meter.result()
    if classifier_meter is not None:
        metrics.update(classifier_meter.result())
    metrics["duration_s"] = time.perf_counter() - start_time
    metrics["data_time_s"] = data_time_s
    metrics["compute_time_s"] = compute_time_s
    metrics["backbone_frames"] = float(backbone_frames)
    metrics["valid_frame_slots"] = float(valid_frame_slots)
    metrics["frame_slots"] = float(frame_slots)
    metrics["frame_reduction_pct"] = 100.0 * (
        1.0 - backbone_frames / max(frame_slots, 1)
    )
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["skipped_optimizer_steps"] = float(skipped_optimizer_steps)
    return metrics


def _set_backbone_trainable(
    model: EdgeFallST | BBoxFallClassifier,
    trainable: bool,
) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(trainable)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: EdgeFallST | BBoxFallClassifier,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "arguments": _serializable(vars(args)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    if best_metric_name == "loss":
        payload["best_val_loss"] = best_metric_value
    return payload


def _load_checkpoint(
    path: Path,
    *,
    model: EdgeFallST | BBoxFallClassifier,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    fallback = (
        math.inf
        if checkpoint.get("arguments", {}).get(
            "training_mode",
            "unified",
        )
        == "unified"
        else float("-inf")
    )
    best_value = checkpoint.get(
        "best_metric_value",
        checkpoint.get("best_val_loss", fallback),
    )
    return int(checkpoint["epoch"]) + 1, float(best_value)


def _initialize_classifier_geometry(
    path: Path,
    model: EdgeFallST | BBoxFallClassifier,
) -> int:
    if not isinstance(model, BBoxFallClassifier):
        raise TypeError("Geometry initialization requires bbox-classifier")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Geometry checkpoint does not contain model state")
    return model.load_geometry_state_dict(state_dict)


def _train_impl(
    args: argparse.Namespace,
    artifacts: RunArtifacts,
    device: torch.device,
) -> dict[str, float]:
    logger = artifacts.logger
    _seed_everything(args.seed, args.deterministic)
    logger.info("run_id=%s", artifacts.run_id)
    logger.info("parameters=%s", artifacts.parameters_path)
    logger.info("checkpoints=%s", artifacts.checkpoint_dir)
    logger.info("device=%s amp_requested=%s", device, args.amp)

    train_dataset, val_dataset = _make_datasets(args)
    train_loader, val_loader, train_sampler = _make_loaders(
        args,
        train_dataset,
        val_dataset,
    )
    logger.info(
        "data train_windows=%d val_windows=%d train_batches=%d val_batches=%d",
        len(train_dataset),
        len(val_dataset),
        len(train_loader),
        len(val_loader),
    )
    if args.training_mode == "bbox-classifier":
        logger.info(
            "event_targets train_positive_videos=%d val_positive_videos=%d "
            "positive_videos_per_batch=%d event_stage_balanced=%s "
            "history_truncation_prob=%.3f",
            sum(train_dataset.video_has_fall_event.values()),
            sum(val_dataset.video_has_fall_event.values()),
            args.positive_videos_per_batch,
            _classifier_event_stage_balanced(args),
            args.history_truncation_prob,
        )
        logger.info(
            "classifier temporal_model=%s spatial_pool=%d "
            "history_offsets=%s control_loss_weight=%.3f "
            "event_state_loss_weight=%.3f geometry_stage_loss_weight=%.3f "
            "geometry_fall_loss_weight=%.3f video_pooling=%s video_topk=%d",
            args.classifier_temporal_model,
            _classifier_spatial_pool_size(args),
            _classifier_history_offsets(args),
            _classifier_control_loss_weight(args),
            _classifier_versioned_loss_weight(
                args,
                "classifier_event_state_loss_weight",
                v33_default=0.5,
            ),
            _classifier_versioned_loss_weight(
                args,
                "classifier_geometry_stage_loss_weight",
                v33_default=0.5,
                v33_g_default=0.0,
            ),
            _classifier_versioned_loss_weight(
                args,
                "classifier_geometry_fall_loss_weight",
                v33_default=1.0,
                v33_g_default=0.0,
            ),
            _classifier_video_pooling(args),
            args.classifier_video_topk,
        )

    model, criterion = _make_model_and_criterion(args)
    geometry_init = getattr(
        args,
        "classifier_geometry_init_checkpoint",
        None,
    )
    if geometry_init is not None:
        loaded_parameters = _initialize_classifier_geometry(
            geometry_init,
            model,
        )
        if args.classifier_freeze_geometry:
            if not isinstance(model, BBoxFallClassifier):
                raise TypeError("Geometry freezing requires bbox-classifier")
            model.freeze_geometry_branch()
        logger.info(
            "geometry_initialized checkpoint=%s parameters=%d frozen=%s",
            geometry_init,
            loaded_parameters,
            args.classifier_freeze_geometry,
        )
    model.to(device)
    criterion.to(device)
    optimizer = _make_optimizer(args, model)
    scheduler = _make_scheduler(args, optimizer)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = (
        torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    )
    if (
        amp_enabled
        and amp_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        logger.warning(
            "BF16 is not supported by this CUDA device; AMP is disabled"
        )
        amp_enabled = False
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
        init_scale=args.initial_loss_scale,
    )
    logger.info(
        "model mode=%s parameters=%d trainable_parameters=%d "
        "amp_enabled=%s amp_dtype=%s",
        args.training_mode,
        model.parameter_count,
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        amp_enabled,
        args.amp_dtype,
    )

    start_epoch = 0
    best_metric_name = (
        "video_mean_precision"
        if args.training_mode == "bbox-classifier"
        else "loss"
    )
    maximize_best_metric = args.training_mode == "bbox-classifier"
    best_metric_value = (
        float("-inf") if maximize_best_metric else math.inf
    )
    if args.resume is not None:
        start_epoch, best_metric_value = _load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        logger.info(
            "resumed checkpoint=%s start_epoch=%d best_%s=%.6f",
            args.resume,
            start_epoch + 1,
            best_metric_name,
            best_metric_value,
        )
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint epoch already reaches requested epochs")

    latest_val_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        backbone_frozen = epoch < args.freeze_backbone_epochs
        _set_backbone_trainable(model, not backbone_frozen)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        logger.info(
            "epoch=%d/%d start backbone_frozen=%s lr=%s",
            epoch + 1,
            args.epochs,
            backbone_frozen,
            ",".join(f"{value:.8g}" for value in learning_rates),
        )
        train_metrics = _run_epoch(
            model=model,
            criterion=criterion,
            training_mode=args.training_mode,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            accumulation_steps=args.accumulation_steps,
            gradient_clip=args.gradient_clip,
            maximum_batches=args.max_train_batches,
            logger=logger,
            log_interval=args.log_interval,
            epoch=epoch,
            backbone_frozen=backbone_frozen,
            history_truncation_prob=args.history_truncation_prob,
            horizontal_flip_prob=args.horizontal_flip_prob,
            grayscale_prob=args.grayscale_prob,
            photometric_jitter=args.photometric_jitter,
            bbox_jitter=args.bbox_jitter,
        )
        logger.info(
            "epoch=%d train_complete %s duration_s=%.1f "
            "data_time_s=%.1f compute_time_s=%.1f "
            "optimizer_steps=%d amp_skips=%d",
            epoch + 1,
            _format_metrics(train_metrics),
            train_metrics["duration_s"],
            train_metrics["data_time_s"],
            train_metrics["compute_time_s"],
            int(train_metrics["optimizer_steps"]),
            int(train_metrics["skipped_optimizer_steps"]),
        )

        should_validate = (
            (epoch + 1) % args.validate_every == 0
            or epoch + 1 == args.epochs
        )
        if should_validate:
            latest_val_metrics = _run_epoch(
                model=model,
                criterion=criterion,
                training_mode=args.training_mode,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                accumulation_steps=1,
                gradient_clip=0.0,
                maximum_batches=args.max_val_batches,
                logger=logger,
                log_interval=args.log_interval,
                epoch=epoch,
                backbone_frozen=False,
                history_truncation_prob=0.0,
                horizontal_flip_prob=0.0,
                grayscale_prob=0.0,
                photometric_jitter=0.0,
                bbox_jitter=0.0,
            )
            logger.info(
                "epoch=%d val_complete %s duration_s=%.1f "
                "data_time_s=%.1f compute_time_s=%.1f",
                epoch + 1,
                _format_metrics(latest_val_metrics),
                latest_val_metrics["duration_s"],
                latest_val_metrics["data_time_s"],
                latest_val_metrics["compute_time_s"],
            )
        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        else:
            logger.warning(
                "epoch=%d scheduler_not_advanced because no optimizer step "
                "completed",
                epoch + 1,
            )

        missing_default = (
            float("-inf") if maximize_best_metric else math.inf
        )
        current_metric_value = latest_val_metrics.get(
            best_metric_name,
            missing_default,
        )
        improved = should_validate and (
            current_metric_value > best_metric_value
            if maximize_best_metric
            else current_metric_value < best_metric_value
        )
        if improved:
            best_metric_value = current_metric_value
        payload = _checkpoint_payload(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            train_metrics=train_metrics,
            val_metrics=latest_val_metrics,
        )
        _atomic_torch_save(payload, artifacts.checkpoint_dir / "last.pt")
        if improved:
            _atomic_torch_save(payload, artifacts.checkpoint_dir / "best.pt")
            logger.info(
                "epoch=%d new_best %s=%.6f",
                epoch + 1,
                best_metric_name,
                best_metric_value,
            )
        if (epoch + 1) % args.save_every == 0:
            _atomic_torch_save(
                payload,
                artifacts.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
            )
        if device.type == "cuda":
            logger.info(
                "epoch=%d cuda_peak_allocated_mb=%.1f "
                "cuda_peak_reserved_mb=%.1f",
                epoch + 1,
                torch.cuda.max_memory_allocated(device) / 1024**2,
                torch.cuda.max_memory_reserved(device) / 1024**2,
            )

    logger.info(
        "training_complete best_%s=%.6f",
        best_metric_name,
        best_metric_value,
    )
    return {
        f"best_{best_metric_name}": best_metric_value,
        **{
            f"val_{name}": value
            for name, value in latest_val_metrics.items()
        },
    }


def run_training(args: argparse.Namespace) -> dict[str, float]:
    _validate_arguments(args)
    device = _resolve_device(args.device)
    artifacts = _create_run_artifacts(args, device=device)
    try:
        return _train_impl(args, artifacts, device)
    except Exception:
        artifacts.logger.exception("training_failed")
        raise
    finally:
        _close_logger(artifacts.logger)


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
