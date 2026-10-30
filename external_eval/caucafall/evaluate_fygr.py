"""Evaluate V3.4-FYGR on CAUCAFall using the author person boxes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import torch

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
from external_eval.fygr_common import (
    FrameItem,
    FygrExternalRuntime,
    evaluate_external_windows,
    extract_external_sequence,
    score_columns,
)


@dataclass
class VideoOutcome:
    sample_id: str
    subject_id: int
    activity: str
    lighting: str
    label: bool
    score: float | None
    geometry_score: float | None
    global_score: float | None
    roi_score: float | None
    status: str
    reason: str
    source_frames: int
    annotated_frames: int
    fall_frames: int
    missing_annotations: int
    sampled_frames: int = 0
    decoded_samples: int = 0
    valid_windows: int = 0
    ground_truth_onset_s: float | None = None
    predicted_onset_s: float | None = None
    alert_delay_s: float | None = None
    duration_s: float = 0.0


@dataclass(frozen=True)
class FramePrediction:
    sample_id: str
    source_index: int
    image_path: str
    timestamp_s: float
    target: bool
    score: float
    geometry_score: float
    global_score: float | None
    roi_score: float | None
    onset_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/caucafall/Dataset CAUCAFall/CAUCAFall"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("yolo26x.pt"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/external_evaluation/caucafall"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--window-batch-size", type=int, default=512)
    parser.add_argument("--limit-per-label", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    for path in (args.checkpoint, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.feature_batch_size, args.window_batch_size, args.log_interval) <= 0:
        raise ValueError("Batch and logging values must be positive")
    if args.limit_per_label < 0 or args.source_fps <= 0:
        raise ValueError("Limits must be non-negative and source FPS positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")


def _empty_outcome(
    record: CaucaVideo, *, status: str, reason: str
) -> VideoOutcome:
    return VideoOutcome(
        sample_id=record.sample_id,
        subject_id=record.subject_id,
        activity=record.activity,
        lighting=record.lighting,
        label=record.label,
        score=None,
        geometry_score=None,
        global_score=None,
        roi_score=None,
        status=status,
        reason=reason,
        source_frames=len(record.frames),
        annotated_frames=record.annotated_frames,
        fall_frames=record.fall_frames,
        missing_annotations=record.missing_annotations,
    )


def _sampled_frames(
    record: CaucaVideo, source_indices: Sequence[int]
) -> Iterator[FrameItem]:
    for sample_index, source_index in enumerate(source_indices):
        frame = record.frames[source_index]
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is not None:
            yield sample_index, image, frame.bbox_xyxy_norm


def _evaluate_record(
    record: CaucaVideo,
    *,
    runtime: FygrExternalRuntime,
    source_fps: float,
    feature_batch_size: int,
    window_batch_size: int,
    threshold: float,
) -> tuple[VideoOutcome, list[FramePrediction]]:
    started = time.perf_counter()
    source_indices = resampled_source_indices(
        len(record.frames), source_fps, runtime.effective_fps
    )
    sequence = extract_external_sequence(
        runtime,
        _sampled_frames(record, source_indices),
        sample_count=len(source_indices),
        feature_batch_size=feature_batch_size,
    )
    targets = [record.frames[index].target for index in source_indices]
    current_indices = [
        index
        for index, target in enumerate(targets)
        if target is not None
        and sequence.decoded_mask[index] > 0
        and sequence.actor_valid_mask[index] > 0
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
    scores = evaluate_external_windows(
        runtime,
        sequence,
        current_indices,
        window_batch_size=window_batch_size,
    )
    columns = score_columns(scores)
    maxima = {name: float(values.max()) for name, values in columns.items()}
    predictions: list[FramePrediction] = []
    for row, sample_index in enumerate(scores.sample_indices):
        source_index = source_indices[sample_index]
        source_frame = record.frames[source_index]
        if source_frame.target is None:
            raise RuntimeError("A valid prediction lost its frame target")
        predictions.append(
            FramePrediction(
                sample_id=record.sample_id,
                source_index=source_index,
                image_path=str(source_frame.image_path.resolve()),
                timestamp_s=source_index / source_fps,
                target=source_frame.target,
                score=float(scores.score[row]),
                geometry_score=float(scores.geometry_score[row]),
                global_score=(
                    None
                    if scores.global_score is None
                    else float(scores.global_score[row])
                ),
                roi_score=(
                    None
                    if scores.roi_score is None
                    else float(scores.roi_score[row])
                ),
                onset_score=float(scores.onset_score[row]),
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
            if prediction.score >= threshold
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
        score=maxima["final"],
        geometry_score=maxima["geometry"],
        global_score=maxima.get("global"),
        roi_score=maxima.get("roi"),
        status="ok",
        reason="",
        source_frames=len(record.frames),
        annotated_frames=record.annotated_frames,
        fall_frames=record.fall_frames,
        missing_annotations=record.missing_annotations,
        sampled_frames=len(source_indices),
        decoded_samples=int(sequence.decoded_mask.sum()),
        valid_windows=len(current_indices),
        ground_truth_onset_s=ground_truth_onset,
        predicted_onset_s=predicted_onset,
        alert_delay_s=delay,
        duration_s=time.perf_counter() - started,
    )
    return outcome, predictions


def _score_block(
    scores: Sequence[float], labels: Sequence[bool], *, threshold: float
) -> dict[str, Any]:
    return {
        "samples": len(scores),
        "positives": sum(int(value) for value in labels),
        "negatives": sum(int(not value) for value in labels),
        "ranking": binary_ranking_metrics(scores, labels),
        "at_fixed_threshold": fixed_threshold_metrics(
            scores, labels, threshold=threshold
        ),
    }


def _metrics(
    outcomes: Sequence[VideoOutcome],
    predictions: Sequence[FramePrediction],
    *,
    threshold: float,
) -> dict[str, Any]:
    evaluable = [row for row in outcomes if row.score is not None]
    result: dict[str, Any] = {
        "coverage": {
            "total_videos": len(outcomes),
            "evaluable_videos": len(evaluable),
            "coverage": len(evaluable) / max(len(outcomes), 1),
            "status_counts": dict(sorted(Counter(row.status for row in outcomes).items())),
            "failure_reason_counts": dict(
                sorted(Counter(row.reason for row in outcomes if row.reason).items())
            ),
            "valid_windows": sum(row.valid_windows for row in outcomes),
        }
    }
    for output_name, video_field, frame_field in (
        ("final", "score", "score"),
        ("geometry", "geometry_score", "geometry_score"),
        ("global", "global_score", "global_score"),
        ("roi", "roi_score", "roi_score"),
    ):
        video_rows = [row for row in outcomes if getattr(row, video_field) is not None]
        frame_rows = [row for row in predictions if getattr(row, frame_field) is not None]
        if not video_rows:
            continue
        result[output_name] = {
            "video": _score_block(
                [float(getattr(row, video_field)) for row in video_rows],
                [row.label for row in video_rows],
                threshold=threshold,
            ),
            "frame": _score_block(
                [float(getattr(row, frame_field)) for row in frame_rows],
                [row.target for row in frame_rows],
                threshold=threshold,
            ),
        }
    return result


def evaluate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = select_balanced_records(
        discover_videos(args.dataset_root), args.limit_per_label
    )
    runtime = FygrExternalRuntime(
        checkpoint_path=args.checkpoint,
        weights_path=args.weights,
        device=device,
        amp=args.amp,
    )
    run_name = args.run_name or (
        "v34_fygr_author_bbox_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "dataset_root": str(args.dataset_root.resolve()),
        "annotation_source": "CAUCAFall author class-plus-person-box TXT files",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": runtime.checkpoint_epoch,
        "weights": str(args.weights.resolve()),
        "feature_contract": runtime.contract,
        "video_score": "maximum window score per video",
        "threshold": float(args.threshold),
        "source_fps": float(args.source_fps),
        "limit_per_label": int(args.limit_per_label),
        "locked_external_test": args.limit_per_label == 0,
        "feature_batch_size": int(args.feature_batch_size),
        "window_batch_size": int(args.window_batch_size),
        "device": str(device),
        "amp": bool(args.amp),
        "parameter_count_without_yolo": runtime.parameter_count_without_yolo,
        "trainable_parameter_count": runtime.trainable_parameter_count,
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outcomes: list[VideoOutcome] = []
    predictions: list[FramePrediction] = []
    started = time.perf_counter()
    try:
        with (output_dir / "predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(VideoOutcome.__annotations__))
            writer.writeheader()
            for index, record in enumerate(records, start=1):
                try:
                    outcome, frame_rows = _evaluate_record(
                        record,
                        runtime=runtime,
                        source_fps=args.source_fps,
                        feature_batch_size=args.feature_batch_size,
                        window_batch_size=args.window_batch_size,
                        threshold=args.threshold,
                    )
                except Exception as error:
                    outcome = _empty_outcome(
                        record,
                        status="error",
                        reason=f"{type(error).__name__}: {error}",
                    )
                    frame_rows = []
                outcomes.append(outcome)
                predictions.extend(frame_rows)
                writer.writerow(asdict(outcome))
                handle.flush()
                if index % args.log_interval == 0 or index == len(records):
                    elapsed = time.perf_counter() - started
                    print(
                        f"videos={index}/{len(records)} "
                        f"ok={sum(row.status == 'ok' for row in outcomes)} "
                        f"videos_per_s={index / max(elapsed, 1e-6):.2f}",
                        flush=True,
                    )
    finally:
        runtime.close()
    with (output_dir / "frame_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    summary = {
        "protocol": protocol,
        "duration_s": time.perf_counter() - started,
        "metrics": _metrics(outcomes, predictions, threshold=args.threshold),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
