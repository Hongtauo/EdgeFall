"""Evaluate a frozen bbox classifier with CAUCAFall author annotations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch
from torch import Tensor

from edgefall_st.data.dataset import (
    _letterbox_image_uint8,
    _transform_normalized_box,
)
from edgefall_st.models.bbox_classifier import BBoxFallClassifier
from external_eval.caucafall.common import (
    SOURCE_FPS,
    CaucaVideo,
    discover_videos,
    select_balanced_records,
)
from external_eval.fall_video_dataset.common import (
    binary_ranking_metrics,
    fixed_threshold_metrics,
    resampled_source_indices,
)
from external_eval.fall_video_dataset.evaluate_v3 import _load_model


VIDEO_FIELDS = (
    "sample_id",
    "subject_id",
    "activity",
    "lighting",
    "label",
    "score",
    "status",
    "reason",
    "source_frames",
    "annotated_frames",
    "fall_frames",
    "alias_pairs",
    "missing_annotations",
    "sampled_frames",
    "valid_windows",
    "ground_truth_onset_s",
    "predicted_onset_s",
    "alert_delay_s",
    "duration_s",
)


@dataclass(frozen=True)
class FramePrediction:
    sample_id: str
    source_index: int
    image_path: str
    timestamp_s: float
    target: bool
    fall_score: float
    stage_probabilities: tuple[float, float, float]
    onset_score: float
    bbox_xyxy_norm: tuple[float, float, float, float]


@dataclass
class VideoOutcome:
    sample_id: str
    subject_id: int
    activity: str
    lighting: str
    label: bool
    score: float | None
    status: str
    reason: str
    source_frames: int
    annotated_frames: int
    fall_frames: int
    alias_pairs: int
    missing_annotations: int
    sampled_frames: int = 0
    valid_windows: int = 0
    ground_truth_onset_s: float | None = None
    predicted_onset_s: float | None = None
    alert_delay_s: float | None = None
    duration_s: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/caucafall/Dataset CAUCAFall/CAUCAFall"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/external_evaluation/caucafall"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-batch-size", type=int, default=96)
    parser.add_argument("--limit-per-label", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.crop_batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("batch size and log interval must be positive")
    if args.limit_per_label < 0:
        raise ValueError("limit-per-label cannot be negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.source_fps <= 0:
        raise ValueError("source-fps must be positive")


def _empty_outcome(
    record: CaucaVideo,
    *,
    status: str,
    reason: str,
) -> VideoOutcome:
    return VideoOutcome(
        sample_id=record.sample_id,
        subject_id=record.subject_id,
        activity=record.activity,
        lighting=record.lighting,
        label=record.label,
        score=None,
        status=status,
        reason=reason,
        source_frames=len(record.frames),
        annotated_frames=record.annotated_frames,
        fall_frames=record.fall_frames,
        alias_pairs=record.alias_pairs,
        missing_annotations=record.missing_annotations,
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
    image_tensor = torch.stack(images).to(device=device, dtype=torch.float32)
    image_tensor.div_(255.0)
    box_tensor = torch.tensor(boxes, device=device, dtype=torch.float32)
    with torch.inference_mode():
        crops = model.cropper(image_tensor, box_tensor)
    crops = crops.mul(255.0).round().clamp_(0.0, 255.0).byte().cpu()
    output[torch.tensor(sample_positions, dtype=torch.long)] = crops


def _history_rows(
    current_indices: Sequence[int],
    history_offsets: Sequence[int],
) -> tuple[list[list[int]], list[list[float]]]:
    rows: list[list[int]] = []
    masks: list[list[float]] = []
    for current in current_indices:
        row: list[int] = []
        mask: list[float] = []
        for offset in history_offsets:
            history_index = current - int(offset)
            row.append(history_index if history_index >= 0 else -1)
            mask.append(float(history_index >= 0))
        rows.append(row)
        masks.append(mask)
    return rows, masks


def _evaluate_video(
    record: CaucaVideo,
    *,
    model: BBoxFallClassifier,
    history_offsets: tuple[int, ...],
    effective_fps: float,
    source_fps: float,
    crop_batch_size: int,
    threshold: float,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[VideoOutcome, list[FramePrediction]]:
    started = time.perf_counter()
    source_indices = resampled_source_indices(
        len(record.frames), source_fps, effective_fps
    )
    sample_count = len(source_indices)
    crop_height, crop_width = model.crop_size
    crops = torch.zeros(
        (sample_count, 3, crop_height, crop_width), dtype=torch.uint8
    )
    boxes_by_sample: list[tuple[float, float, float, float] | None] = [
        None
    ] * sample_count
    targets_by_sample: list[bool | None] = [None] * sample_count
    paths_by_sample: list[Path | None] = [None] * sample_count
    actor_valid = torch.zeros(sample_count, dtype=torch.float32)
    pending_images: list[Tensor] = []
    pending_boxes: list[tuple[float, float, float, float]] = []
    pending_positions: list[int] = []

    for sample_index, source_index in enumerate(source_indices):
        frame = record.frames[source_index]
        targets_by_sample[sample_index] = frame.target
        paths_by_sample[sample_index] = frame.image_path
        if frame.bbox_xyxy_norm is None:
            continue
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        source_h, source_w = image.shape[:2]
        image_tensor, scale, pad_left, pad_top = _letterbox_image_uint8(
            image,
            model.input_size,
            fill_value=114,
        )
        transformed = _transform_normalized_box(
            list(frame.bbox_xyxy_norm),
            source_size=(source_h, source_w),
            input_size=model.input_size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        if transformed is None:
            continue
        box = tuple(float(value) for value in transformed)
        boxes_by_sample[sample_index] = box
        actor_valid[sample_index] = 1.0
        pending_images.append(image_tensor)
        pending_boxes.append(box)
        pending_positions.append(sample_index)
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

    current_indices = [
        index
        for index, target in enumerate(targets_by_sample)
        if target is not None and actor_valid[index] > 0
    ]
    if not current_indices:
        return (
            _empty_outcome(
                record,
                status="skipped",
                reason="no_valid_sampled_author_boxes",
            ),
            [],
        )
    history_rows, valid_rows = _history_rows(current_indices, history_offsets)
    box_values = [
        box if box is not None else (0.0, 0.0, 0.0, 0.0)
        for box in boxes_by_sample
    ]
    images = crops.to(device=device, dtype=torch.float32).div_(255.0)
    boxes = torch.tensor(box_values, device=device, dtype=torch.float32)
    history_indices = torch.tensor(
        history_rows, device=device, dtype=torch.long
    )
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
    stage_probabilities = outputs["stage_logits"].float().softmax(-1).cpu()
    onset_scores = outputs["onset_logits"].float().sigmoid().cpu()
    predictions: list[FramePrediction] = []
    for output_index, sample_index in enumerate(current_indices):
        source_index = source_indices[sample_index]
        target = targets_by_sample[sample_index]
        image_path = paths_by_sample[sample_index]
        box = boxes_by_sample[sample_index]
        if target is None or image_path is None or box is None:
            raise RuntimeError("A valid output lost its source annotation")
        predictions.append(
            FramePrediction(
                sample_id=record.sample_id,
                source_index=source_index,
                image_path=str(image_path.resolve()),
                timestamp_s=source_index / source_fps,
                target=target,
                fall_score=float(fall_scores[output_index]),
                stage_probabilities=tuple(
                    float(value)
                    for value in stage_probabilities[output_index].tolist()
                ),
                onset_score=float(onset_scores[output_index]),
                bbox_xyxy_norm=record.frames[source_index].bbox_xyxy_norm,
            )
        )

    ground_truth_onset = next(
        (
            frame.source_index / source_fps
            for frame in record.frames
            if frame.target is True
        ),
        None,
    )
    predicted_onset = next(
        (
            prediction.timestamp_s
            for prediction in predictions
            if prediction.fall_score >= threshold
        ),
        None,
    )
    delay = (
        None
        if ground_truth_onset is None or predicted_onset is None
        else predicted_onset - ground_truth_onset
    )
    outcome = VideoOutcome(
        sample_id=record.sample_id,
        subject_id=record.subject_id,
        activity=record.activity,
        lighting=record.lighting,
        label=record.label,
        score=max(prediction.fall_score for prediction in predictions),
        status="ok",
        reason="",
        source_frames=len(record.frames),
        annotated_frames=record.annotated_frames,
        fall_frames=record.fall_frames,
        alias_pairs=record.alias_pairs,
        missing_annotations=record.missing_annotations,
        sampled_frames=sample_count,
        valid_windows=len(predictions),
        ground_truth_onset_s=ground_truth_onset,
        predicted_onset_s=predicted_onset,
        alert_delay_s=delay,
        duration_s=time.perf_counter() - started,
    )
    return outcome, predictions


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _score_block(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    threshold: float,
) -> dict[str, Any]:
    return {
        "samples": len(scores),
        "positives": sum(int(label) for label in labels),
        "negatives": sum(int(not label) for label in labels),
        "ranking": binary_ranking_metrics(scores, labels),
        "at_fixed_threshold": fixed_threshold_metrics(
            scores, labels, threshold=threshold
        ),
    }


def _stratified_video_metrics(
    outcomes: Sequence[VideoOutcome],
    *,
    key: str,
    threshold: float,
) -> dict[str, Any]:
    groups: dict[str, list[VideoOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.score is not None:
            groups[str(getattr(outcome, key))].append(outcome)
    result: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        scores = [float(row.score) for row in rows if row.score is not None]
        labels = [row.label for row in rows]
        result[name] = {
            **_score_block(scores, labels, threshold=threshold),
            "score_distribution": _distribution(scores),
        }
    return result


def _metrics(
    outcomes: Sequence[VideoOutcome],
    frame_predictions: Sequence[FramePrediction],
    *,
    threshold: float,
) -> dict[str, Any]:
    evaluable = [outcome for outcome in outcomes if outcome.score is not None]
    video_scores = [float(outcome.score) for outcome in evaluable]
    video_labels = [outcome.label for outcome in evaluable]
    frame_scores = [prediction.fall_score for prediction in frame_predictions]
    frame_labels = [prediction.target for prediction in frame_predictions]
    status_counts = Counter(outcome.status for outcome in outcomes)
    reason_counts = Counter(
        outcome.reason for outcome in outcomes if outcome.reason
    )
    delays = [
        float(outcome.alert_delay_s)
        for outcome in evaluable
        if outcome.label and outcome.alert_delay_s is not None
    ]
    return {
        "coverage": {
            "total_videos": len(outcomes),
            "evaluable_videos": len(evaluable),
            "coverage": len(evaluable) / max(len(outcomes), 1),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_reason_counts": dict(sorted(reason_counts.items())),
            "source_frames": sum(outcome.source_frames for outcome in outcomes),
            "annotated_frames": sum(
                outcome.annotated_frames for outcome in outcomes
            ),
            "missing_annotations": sum(
                outcome.missing_annotations for outcome in outcomes
            ),
            "numeric_alias_pairs": sum(outcome.alias_pairs for outcome in outcomes),
            "valid_windows": sum(outcome.valid_windows for outcome in outcomes),
        },
        "video": _score_block(video_scores, video_labels, threshold=threshold),
        "frame": _score_block(frame_scores, frame_labels, threshold=threshold),
        "event_timing_at_fixed_threshold": {
            "detected_positive_videos": len(delays),
            "early_alerts": sum(delay < 0.0 for delay in delays),
            "delay_s": _distribution(delays),
        },
        "by_activity": _stratified_video_metrics(
            evaluable, key="activity", threshold=threshold
        ),
        "by_lighting": _stratified_video_metrics(
            evaluable, key="lighting", threshold=threshold
        ),
        "by_subject": _stratified_video_metrics(
            evaluable, key="subject_id", threshold=threshold
        ),
    }


def evaluate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    all_records = discover_videos(args.dataset_root)
    records = select_balanced_records(all_records, args.limit_per_label)
    model, saved, offsets, amp_enabled, amp_dtype = _load_model(
        args.checkpoint, device
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.csv"
    frame_predictions_path = output_dir / "frame_predictions.jsonl"
    summary_path = output_dir / "summary.json"
    config = {
        "dataset_root": str(args.dataset_root.resolve()),
        "annotation_source": "CAUCAFall author class-plus-person-box TXT files",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "device": str(device),
        "model_parameter_count": model.parameter_count,
        "temporal_model": getattr(saved, "classifier_temporal_model", "attention"),
        "history_offsets": list(offsets),
        "source_fps": float(args.source_fps),
        "effective_fps": float(saved.effective_fps),
        "input_size": list(model.input_size),
        "crop_size": list(model.crop_size),
        "crop_context": float(model.cropper.context_scale),
        "video_score": "maximum sigmoid(fall_logits) across valid sampled frames",
        "threshold": float(args.threshold),
        "limit_per_label": int(args.limit_per_label),
        "locked_external_test": args.limit_per_label == 0,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    outcomes: list[VideoOutcome] = []
    all_frame_predictions: list[FramePrediction] = []
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_FIELDS)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            try:
                outcome, predictions = _evaluate_video(
                    record,
                    model=model,
                    history_offsets=offsets,
                    effective_fps=float(saved.effective_fps),
                    source_fps=float(args.source_fps),
                    crop_batch_size=args.crop_batch_size,
                    threshold=float(args.threshold),
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            except Exception as error:
                outcome = _empty_outcome(
                    record,
                    status="error",
                    reason=f"{type(error).__name__}: {error}",
                )
                predictions = []
            outcomes.append(outcome)
            all_frame_predictions.extend(predictions)
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

    with frame_predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in all_frame_predictions:
            handle.write(
                json.dumps(asdict(prediction), ensure_ascii=False) + "\n"
            )
    summary = {
        "protocol": config,
        "duration_s": time.perf_counter() - started,
        "metrics": _metrics(
            outcomes,
            all_frame_predictions,
            threshold=float(args.threshold),
        ),
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
