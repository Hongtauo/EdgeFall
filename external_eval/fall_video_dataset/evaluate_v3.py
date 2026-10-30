"""Evaluate a frozen V3 checkpoint on the external Fall/No_Fall dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import torch
from torch import Tensor

from edgefall_st.data.dataset import (
    _letterbox_image_uint8,
    _letterbox_parameters,
    _transform_normalized_box,
)
from edgefall_st.models.bbox_classifier import (
    BBoxFallClassifier,
    V32_LONG_HISTORY_OFFSETS,
    V32_SHORT_HISTORY_OFFSETS,
)
from external_eval.fall_video_dataset.common import (
    VideoRecord,
    binary_ranking_metrics,
    discover_records,
    fixed_threshold_metrics,
    read_pose_boxes,
    resampled_source_indices,
)


PREDICTION_FIELDS = (
    "sample_id",
    "group_id",
    "label_name",
    "label",
    "score",
    "status",
    "reason",
    "video_path",
    "keypoints_path",
    "source_frames",
    "source_fps",
    "width",
    "height",
    "pose_frames",
    "valid_pose_frames",
    "effective_samples",
    "valid_samples",
    "windows",
    "frame_alignment_delta",
    "duration_s",
)


@dataclass
class VideoOutcome:
    sample_id: str
    group_id: str
    label_name: str
    label: bool
    score: float | None
    status: str
    reason: str
    video_path: str
    keypoints_path: str
    source_frames: int = 0
    source_fps: float = 0.0
    width: int = 0
    height: int = 0
    pose_frames: int = 0
    valid_pose_frames: int = 0
    effective_samples: int = 0
    valid_samples: int = 0
    windows: int = 0
    frame_alignment_delta: int = 0
    duration_s: float = 0.0


@dataclass(frozen=True)
class NormalizedBoxSequence:
    boxes: dict[int, tuple[float, float, float, float]]
    source_path: Path
    observed_frames: int
    valid_frames: int
    frame_alignment_delta: int
    no_valid_reason: str


@dataclass(frozen=True)
class ExternalWindowPrediction:
    sample_index: int
    timestamp_s: float
    fall_score: float
    stage_probabilities: tuple[float, float, float]
    onset_score: float


BoxLoader = Callable[[int, int, int], NormalizedBoxSequence]
WindowPredictionObserver = Callable[
    [Sequence[ExternalWindowPrediction]],
    None,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/fall-video-dataset"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/external_evaluation/fall_video_dataset"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-keypoints", type=int, default=4)
    parser.add_argument("--crop-batch-size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0.0 <= args.pose_confidence <= 1.0:
        raise ValueError("pose-confidence must be in [0, 1]")
    if args.minimum_keypoints <= 0 or args.crop_batch_size <= 0:
        raise ValueError("minimum-keypoints and crop-batch-size must be positive")
    if args.limit_per_class < 0 or args.log_interval <= 0:
        raise ValueError("limits must be non-negative and log-interval positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")


def _history_offsets(saved: argparse.Namespace) -> tuple[int, ...]:
    if getattr(saved, "classifier_temporal_model", "attention") in {
        "v32",
        "v33-g",
        "v33-gr",
    }:
        short_offsets = getattr(
            saved,
            "classifier_v32_short_offsets",
            V32_SHORT_HISTORY_OFFSETS,
        )
        long_offsets = getattr(
            saved,
            "classifier_v32_long_offsets",
            V32_LONG_HISTORY_OFFSETS,
        )
        return tuple(
            sorted(
                {int(value) for value in short_offsets}.union(
                    int(value) for value in long_offsets
                )
            )
        )
    values = {
        int(offset)
        for name in ("history_p3", "history_p4", "history_p5")
        for offset in getattr(saved, name)
    }
    return tuple(sorted(values))


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BBoxFallClassifier, argparse.Namespace, tuple[int, ...], bool, torch.dtype]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    saved = argparse.Namespace(**checkpoint["arguments"])
    if getattr(saved, "training_mode", None) != "bbox-classifier":
        raise ValueError("The checkpoint is not a bbox-classifier model")
    offsets = _history_offsets(saved)
    temporal_model = getattr(saved, "classifier_temporal_model", "attention")
    spatial_pool_size = getattr(saved, "classifier_spatial_pool_size", None)
    if spatial_pool_size is None:
        spatial_pool_size = (
            2 if temporal_model in {"v32", "v33-gr"} else 1
        )
    model = BBoxFallClassifier(
        input_size=(saved.input_height, saved.input_width),
        crop_size=(saved.crop_height, saved.crop_width),
        crop_context=saved.crop_context,
        backbone_channels=tuple(saved.backbone_channels),
        visual_dim=saved.classifier_visual_dim,
        temporal_dim=saved.classifier_temporal_dim,
        history_offsets=offsets,
        activation=saved.activation,
        temporal_model=temporal_model,
        temporal_dropout=getattr(saved, "classifier_temporal_dropout", 0.1),
        short_horizon=getattr(saved, "classifier_short_horizon", 4),
        spatial_pool_size=int(spatial_pool_size),
        short_history_offsets=getattr(
            saved,
            "classifier_v32_short_offsets",
            V32_SHORT_HISTORY_OFFSETS,
        ),
        long_history_offsets=getattr(
            saved,
            "classifier_v32_long_offsets",
            V32_LONG_HISTORY_OFFSETS,
        ),
        effective_fps=saved.effective_fps,
        pre_cropped_inputs=True,
        visual_modality_dropout=getattr(
            saved,
            "classifier_visual_modality_dropout",
            0.3,
        ),
        visual_gate_bias=getattr(
            saved,
            "classifier_visual_gate_bias",
            -2.0,
        ),
        visual_input_size=getattr(
            saved,
            "classifier_visual_input_size",
            None,
        ),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    amp_dtype = (
        torch.float16
        if getattr(saved, "amp_dtype", "float16") == "float16"
        else torch.bfloat16
    )
    amp_enabled = bool(getattr(saved, "amp", False)) and device.type == "cuda"
    if amp_dtype == torch.bfloat16 and device.type == "cuda":
        amp_enabled = amp_enabled and torch.cuda.is_bf16_supported()
    return model, saved, offsets, amp_enabled, amp_dtype


def _empty_outcome(
    record: VideoRecord,
    *,
    status: str,
    reason: str,
    boxes_path: Path | None = None,
) -> VideoOutcome:
    return VideoOutcome(
        sample_id=record.sample_id,
        group_id=record.group_id,
        label_name=record.label_name,
        label=record.label,
        score=None,
        status=status,
        reason=reason,
        video_path=str(record.video_path.resolve()),
        keypoints_path=str((boxes_path or record.keypoints_path).resolve()),
    )


def _crop_pending(
    model: BBoxFallClassifier,
    images: list[Tensor],
    boxes: list[tuple[float, float, float, float]],
    sample_positions: list[int],
    output: Tensor,
    device: torch.device,
) -> None:
    if not images:
        return
    image_tensor = (
        torch.stack(images)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    box_tensor = torch.tensor(boxes, device=device, dtype=torch.float32)
    crops = model.cropper(image_tensor, box_tensor)
    crops = crops.mul(255.0).round().clamp_(0.0, 255.0).byte().cpu()
    output[torch.tensor(sample_positions, dtype=torch.long)] = crops


def _evaluate_video_with_box_loader(
    record: VideoRecord,
    *,
    model: BBoxFallClassifier,
    history_offsets: tuple[int, ...],
    effective_fps: float,
    box_loader: BoxLoader,
    crop_batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    window_observer: WindowPredictionObserver | None = None,
) -> VideoOutcome:
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(record.video_path))
    if not capture.isOpened():
        return _empty_outcome(record, status="skipped", reason="video_open_failed")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or source_fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        return _empty_outcome(record, status="skipped", reason="invalid_video_metadata")

    box_sequence = box_loader(width, height, frame_count)
    source_indices = resampled_source_indices(frame_count, source_fps, effective_fps)
    sample_count = len(source_indices)
    scale, pad_left, pad_top = _letterbox_parameters(
        (height, width),
        model.input_size,
    )
    boxes_by_sample: list[tuple[float, float, float, float] | None] = []
    for source_index in source_indices:
        source_box = box_sequence.boxes.get(source_index)
        transformed = _transform_normalized_box(
            None if source_box is None else list(source_box),
            source_size=(height, width),
            input_size=model.input_size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        boxes_by_sample.append(
            None if transformed is None else tuple(transformed)
        )
    valid_samples = [box is not None for box in boxes_by_sample]
    if not any(valid_samples):
        capture.release()
        outcome = _empty_outcome(
            record,
            status="skipped",
            reason=box_sequence.no_valid_reason,
            boxes_path=box_sequence.source_path,
        )
        outcome.source_frames = frame_count
        outcome.source_fps = source_fps
        outcome.width = width
        outcome.height = height
        outcome.pose_frames = box_sequence.observed_frames
        outcome.valid_pose_frames = box_sequence.valid_frames
        outcome.effective_samples = sample_count
        outcome.frame_alignment_delta = box_sequence.frame_alignment_delta
        outcome.duration_s = time.perf_counter() - started
        return outcome

    crop_height, crop_width = model.crop_size
    crops = torch.zeros(
        (sample_count, 3, crop_height, crop_width),
        dtype=torch.uint8,
    )
    source_to_samples: dict[int, list[int]] = {}
    for sample_index, (source_index, valid) in enumerate(
        zip(source_indices, valid_samples, strict=True)
    ):
        if valid:
            source_to_samples.setdefault(source_index, []).append(sample_index)

    pending_images: list[Tensor] = []
    pending_boxes: list[tuple[float, float, float, float]] = []
    pending_positions: list[int] = []
    maximum_needed = max(source_to_samples)
    decoded_positions: set[int] = set()
    try:
        for source_index in range(maximum_needed + 1):
            ok, frame = capture.read()
            if not ok:
                break
            for sample_index in source_to_samples.get(source_index, ()):
                box = boxes_by_sample[sample_index]
                if box is None:
                    continue
                image, _, _, _ = _letterbox_image_uint8(
                    frame,
                    model.input_size,
                    fill_value=114,
                )
                pending_images.append(image)
                pending_boxes.append(box)
                pending_positions.append(sample_index)
                decoded_positions.add(sample_index)
                if len(pending_images) >= crop_batch_size:
                    _crop_pending(
                        model,
                        pending_images,
                        pending_boxes,
                        pending_positions,
                        crops,
                        device,
                    )
                    pending_images.clear()
                    pending_boxes.clear()
                    pending_positions.clear()
        _crop_pending(
            model,
            pending_images,
            pending_boxes,
            pending_positions,
            crops,
            device,
        )
    finally:
        capture.release()

    actor_valid = torch.tensor(
        [index in decoded_positions for index in range(sample_count)],
        dtype=torch.float32,
    )
    current_indices = [
        index for index in range(sample_count) if actor_valid[index] > 0
    ]
    if not current_indices:
        return _empty_outcome(record, status="skipped", reason="video_decode_failed")

    history_rows: list[list[int]] = []
    valid_rows: list[list[float]] = []
    for current in current_indices:
        indices: list[int] = []
        valid: list[float] = []
        for offset in history_offsets:
            history_index = current - offset
            indices.append(history_index if history_index >= 0 else -1)
            valid.append(float(history_index >= 0))
        history_rows.append(indices)
        valid_rows.append(valid)

    box_values = [box if box is not None else (0.0, 0.0, 0.0, 0.0) for box in boxes_by_sample]
    images = crops.to(device=device, dtype=torch.float32).div_(255.0)
    boxes = torch.tensor(box_values, device=device, dtype=torch.float32)
    history_indices = torch.tensor(history_rows, device=device, dtype=torch.long)
    valid_mask = torch.tensor(valid_rows, device=device, dtype=torch.float32)
    actor_valid = actor_valid.to(device=device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        outputs = model.forward_indexed_history(
            images,
            boxes,
            history_indices,
            valid_mask,
            actor_valid,
        )
    fall_scores = outputs["fall_logits"].float().sigmoid().cpu()
    score = float(fall_scores.max())
    if window_observer is not None:
        stage_probabilities = outputs["stage_logits"].float().softmax(-1).cpu()
        onset_scores = outputs["onset_logits"].float().sigmoid().cpu()
        window_observer(
            tuple(
                ExternalWindowPrediction(
                    sample_index=current,
                    timestamp_s=current / effective_fps,
                    fall_score=float(fall_scores[position]),
                    stage_probabilities=tuple(
                        float(value)
                        for value in stage_probabilities[position].tolist()
                    ),
                    onset_score=float(onset_scores[position]),
                )
                for position, current in enumerate(current_indices)
            )
        )
    return VideoOutcome(
        sample_id=record.sample_id,
        group_id=record.group_id,
        label_name=record.label_name,
        label=record.label,
        score=score,
        status="ok",
        reason="",
        video_path=str(record.video_path.resolve()),
        keypoints_path=str(box_sequence.source_path.resolve()),
        source_frames=frame_count,
        source_fps=source_fps,
        width=width,
        height=height,
        pose_frames=box_sequence.observed_frames,
        valid_pose_frames=box_sequence.valid_frames,
        effective_samples=sample_count,
        valid_samples=len(decoded_positions),
        windows=len(current_indices),
        frame_alignment_delta=box_sequence.frame_alignment_delta,
        duration_s=time.perf_counter() - started,
    )


def _evaluate_video(
    record: VideoRecord,
    *,
    model: BBoxFallClassifier,
    history_offsets: tuple[int, ...],
    effective_fps: float,
    pose_confidence: float,
    minimum_keypoints: int,
    crop_batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> VideoOutcome:
    def load_pose_boxes(
        width: int,
        height: int,
        frame_count: int,
    ) -> NormalizedBoxSequence:
        pose = read_pose_boxes(
            record.keypoints_path,
            frame_width=width,
            frame_height=height,
            confidence_threshold=pose_confidence,
            minimum_keypoints=minimum_keypoints,
        )
        return NormalizedBoxSequence(
            boxes={index - 1: box for index, box in pose.boxes.items()},
            source_path=record.keypoints_path,
            observed_frames=pose.observed_frames,
            valid_frames=pose.valid_frames,
            frame_alignment_delta=pose.maximum_frame_number - frame_count,
            no_valid_reason="no_valid_pose_boxes",
        )

    return _evaluate_video_with_box_loader(
        record,
        model=model,
        history_offsets=history_offsets,
        effective_fps=effective_fps,
        box_loader=load_pose_boxes,
        crop_batch_size=crop_batch_size,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )


def _metrics(
    outcomes: Sequence[VideoOutcome],
    *,
    threshold: float,
) -> dict[str, Any]:
    evaluable = [outcome for outcome in outcomes if outcome.score is not None]
    evaluable_scores = [float(outcome.score) for outcome in evaluable]
    evaluable_labels = [outcome.label for outcome in evaluable]
    conservative_scores = [
        0.0 if outcome.score is None else float(outcome.score)
        for outcome in outcomes
    ]
    conservative_labels = [outcome.label for outcome in outcomes]

    def metric_block(scores: list[float], labels: list[bool]) -> dict[str, Any]:
        return {
            "videos": len(scores),
            "positives": sum(int(label) for label in labels),
            "negatives": sum(int(not label) for label in labels),
            "ranking": binary_ranking_metrics(scores, labels),
            "at_fixed_threshold": fixed_threshold_metrics(
                scores,
                labels,
                threshold=threshold,
            ),
        }

    status_counts = Counter(outcome.status for outcome in outcomes)
    reason_counts = Counter(
        outcome.reason for outcome in outcomes if outcome.reason
    )
    return {
        "coverage": {
            "total_videos": len(outcomes),
            "evaluable_videos": len(evaluable),
            "coverage": len(evaluable) / max(len(outcomes), 1),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_reason_counts": dict(sorted(reason_counts.items())),
            "frame_alignment_mismatch_videos": sum(
                int(outcome.frame_alignment_delta != 0)
                for outcome in outcomes
                if outcome.source_frames > 0
            ),
        },
        "evaluable_subset": metric_block(evaluable_scores, evaluable_labels),
        "conservative_all_videos": metric_block(
            conservative_scores,
            conservative_labels,
        ),
        "conservative_policy": "assign score 0.0 to every unevaluable video",
    }


def _select_records(
    records: Sequence[VideoRecord],
    limit_per_class: int,
) -> list[VideoRecord]:
    if limit_per_class <= 0:
        return list(records)
    selected: list[VideoRecord] = []
    counts: Counter[str] = Counter()
    for record in records:
        if counts[record.label_name] >= limit_per_class:
            continue
        selected.append(record)
        counts[record.label_name] += 1
    return selected


def evaluate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    records = _select_records(
        discover_records(args.dataset_root),
        args.limit_per_class,
    )
    model, saved, offsets, amp_enabled, amp_dtype = _load_model(
        args.checkpoint,
        device,
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    config = {
        "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(
            torch.load(
                args.checkpoint,
                map_location="cpu",
                weights_only=False,
            )["epoch"]
        )
        + 1,
        "device": str(device),
        "model_parameter_count": model.parameter_count,
        "history_offsets": list(offsets),
        "effective_fps": float(saved.effective_fps),
        "crop_size": list(model.crop_size),
        "crop_context": float(model.cropper.context_scale),
        "pose_confidence": args.pose_confidence,
        "minimum_keypoints": args.minimum_keypoints,
        "video_score": "maximum sigmoid(fall_logits) across valid windows",
        "threshold": args.threshold,
        "limit_per_class": args.limit_per_class,
        "locked_external_test": args.limit_per_class == 0,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    outcomes: list[VideoOutcome] = []
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            try:
                outcome = _evaluate_video(
                    record,
                    model=model,
                    history_offsets=offsets,
                    effective_fps=float(saved.effective_fps),
                    pose_confidence=args.pose_confidence,
                    minimum_keypoints=args.minimum_keypoints,
                    crop_batch_size=args.crop_batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            except Exception as error:  # keep the locked evaluation auditable
                outcome = _empty_outcome(
                    record,
                    status="error",
                    reason=f"{type(error).__name__}: {error}",
                )
            outcomes.append(outcome)
            writer.writerow(asdict(outcome))
            handle.flush()
            if index % args.log_interval == 0 or index == len(records):
                elapsed = time.perf_counter() - started
                ok_count = sum(item.status == "ok" for item in outcomes)
                print(
                    f"videos={index}/{len(records)} ok={ok_count} "
                    f"videos_per_s={index / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )

    summary = {
        "protocol": config,
        "duration_s": time.perf_counter() - started,
        "metrics": _metrics(outcomes, threshold=args.threshold),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
