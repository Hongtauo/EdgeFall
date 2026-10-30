"""Evaluate V3.4-FYGR on Fall Video Dataset with frozen YOLO features."""

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

from external_eval.fall_video_dataset.common import (
    VideoRecord,
    binary_ranking_metrics,
    fixed_threshold_metrics,
    resampled_source_indices,
)
from external_eval.fall_video_dataset.evaluate_v3_yolo import (
    _artifact_path,
    _manifest_records,
    read_actor_box_sequence,
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
    group_id: str
    label_name: str
    label: bool
    score: float | None
    geometry_score: float | None
    global_score: float | None
    roi_score: float | None
    status: str
    reason: str
    video_path: str
    boxes_path: str
    source_frames: int = 0
    source_fps: float = 0.0
    width: int = 0
    height: int = 0
    box_frames: int = 0
    valid_box_frames: int = 0
    effective_samples: int = 0
    decoded_samples: int = 0
    valid_samples: int = 0
    windows: int = 0
    duration_s: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("Dataset/fall-video-dataset")
    )
    parser.add_argument(
        "--boxes-root",
        type=Path,
        default=Path(
            "artifacts/external_pseudo_labels/fall_video_dataset/yolo26x_nms"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("yolo26x.pt"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/external_evaluation/fall_video_dataset"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--window-batch-size", type=int, default=512)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for path in (args.dataset_root, args.boxes_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (args.checkpoint, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.feature_batch_size, args.window_batch_size, args.log_interval) <= 0:
        raise ValueError("Batch and logging values must be positive")
    if args.limit_per_class < 0:
        raise ValueError("limit-per-class cannot be negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")


def _empty_outcome(
    record: VideoRecord,
    boxes_path: Path,
    *,
    status: str,
    reason: str,
) -> VideoOutcome:
    return VideoOutcome(
        sample_id=record.sample_id,
        group_id=record.group_id,
        label_name=record.label_name,
        label=record.label,
        score=None,
        geometry_score=None,
        global_score=None,
        roi_score=None,
        status=status,
        reason=reason,
        video_path=str(record.video_path.resolve()),
        boxes_path=str(boxes_path.resolve()),
    )


def _sampled_frames(
    capture: cv2.VideoCapture,
    source_indices: Sequence[int],
    boxes: dict[int, tuple[float, float, float, float]],
) -> Iterator[FrameItem]:
    positions_by_source: dict[int, list[int]] = {}
    for sample_index, source_index in enumerate(source_indices):
        positions_by_source.setdefault(int(source_index), []).append(sample_index)
    maximum_needed = max(positions_by_source)
    for source_index in range(maximum_needed + 1):
        ok, frame = capture.read()
        if not ok:
            break
        for sample_index in positions_by_source.get(source_index, ()):
            yield sample_index, frame, boxes.get(source_index)


def _evaluate_record(
    record: VideoRecord,
    *,
    boxes_root: Path,
    runtime: FygrExternalRuntime,
    feature_batch_size: int,
    window_batch_size: int,
) -> VideoOutcome:
    started = time.perf_counter()
    boxes_path = _artifact_path(
        boxes_root / "actor_boxes", record.sample_id, ".jsonl"
    )
    capture = cv2.VideoCapture(str(record.video_path))
    if not capture.isOpened():
        return _empty_outcome(
            record, boxes_path, status="skipped", reason="video_open_failed"
        )
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(frame_count, source_fps, width, height) <= 0:
        capture.release()
        return _empty_outcome(
            record, boxes_path, status="skipped", reason="invalid_video_metadata"
        )
    box_sequence = read_actor_box_sequence(boxes_path, frame_count=frame_count)
    source_indices = resampled_source_indices(
        frame_count, source_fps, runtime.effective_fps
    )
    try:
        sequence = extract_external_sequence(
            runtime,
            _sampled_frames(capture, source_indices, box_sequence.boxes),
            sample_count=len(source_indices),
            feature_batch_size=feature_batch_size,
        )
    finally:
        capture.release()
    current_indices = torch.nonzero(
        sequence.actor_valid_mask.gt(0) & sequence.decoded_mask.gt(0),
        as_tuple=False,
    ).flatten().tolist()
    if not current_indices:
        outcome = _empty_outcome(
            record,
            boxes_path,
            status="skipped",
            reason="no_valid_decoded_actor_boxes",
        )
    else:
        scores = evaluate_external_windows(
            runtime,
            sequence,
            current_indices,
            window_batch_size=window_batch_size,
        )
        maxima = {
            name: float(values.max())
            for name, values in score_columns(scores).items()
        }
        outcome = VideoOutcome(
            sample_id=record.sample_id,
            group_id=record.group_id,
            label_name=record.label_name,
            label=record.label,
            score=maxima["final"],
            geometry_score=maxima["geometry"],
            global_score=maxima.get("global"),
            roi_score=maxima.get("roi"),
            status="ok",
            reason="",
            video_path=str(record.video_path.resolve()),
            boxes_path=str(boxes_path.resolve()),
        )
    outcome.source_frames = frame_count
    outcome.source_fps = source_fps
    outcome.width = width
    outcome.height = height
    outcome.box_frames = box_sequence.observed_frames
    outcome.valid_box_frames = box_sequence.valid_frames
    outcome.effective_samples = len(source_indices)
    outcome.decoded_samples = int(sequence.decoded_mask.sum())
    outcome.valid_samples = int(sequence.actor_valid_mask.sum())
    outcome.windows = len(current_indices)
    outcome.duration_s = time.perf_counter() - started
    return outcome


def _metric_block(
    outcomes: Sequence[VideoOutcome],
    *,
    score_name: str,
    threshold: float,
    conservative: bool,
) -> dict[str, Any]:
    rows = list(outcomes) if conservative else [
        row for row in outcomes if getattr(row, score_name) is not None
    ]
    scores = [
        0.0 if getattr(row, score_name) is None else float(getattr(row, score_name))
        for row in rows
    ]
    labels = [row.label for row in rows]
    return {
        "videos": len(rows),
        "positives": sum(int(value) for value in labels),
        "negatives": sum(int(not value) for value in labels),
        "ranking": binary_ranking_metrics(scores, labels),
        "at_fixed_threshold": fixed_threshold_metrics(
            scores, labels, threshold=threshold
        ),
    }


def _metrics(
    outcomes: Sequence[VideoOutcome], *, threshold: float
) -> dict[str, Any]:
    status_counts = Counter(row.status for row in outcomes)
    reason_counts = Counter(row.reason for row in outcomes if row.reason)
    result: dict[str, Any] = {
        "coverage": {
            "total_videos": len(outcomes),
            "evaluable_videos": sum(row.score is not None for row in outcomes),
            "coverage": sum(row.score is not None for row in outcomes)
            / max(len(outcomes), 1),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_reason_counts": dict(sorted(reason_counts.items())),
        }
    }
    for output_name, field_name in (
        ("final", "score"),
        ("geometry", "geometry_score"),
        ("global", "global_score"),
        ("roi", "roi_score"),
    ):
        if any(getattr(row, field_name) is not None for row in outcomes):
            result[output_name] = {
                "evaluable_subset": _metric_block(
                    outcomes,
                    score_name=field_name,
                    threshold=threshold,
                    conservative=False,
                ),
                "conservative_all_videos": _metric_block(
                    outcomes,
                    score_name=field_name,
                    threshold=threshold,
                    conservative=True,
                ),
            }
    return result


def _select_records(
    records: Sequence[VideoRecord], limit_per_class: int
) -> list[VideoRecord]:
    if limit_per_class <= 0:
        return list(records)
    counts: Counter[str] = Counter()
    selected: list[VideoRecord] = []
    for record in records:
        if counts[record.label_name] >= limit_per_class:
            continue
        counts[record.label_name] += 1
        selected.append(record)
    return selected


def evaluate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = _select_records(
        _manifest_records(args.dataset_root, args.boxes_root),
        args.limit_per_class,
    )
    runtime = FygrExternalRuntime(
        checkpoint_path=args.checkpoint,
        weights_path=args.weights,
        device=device,
        amp=args.amp,
    )
    run_name = args.run_name or (
        "v34_fygr_yolo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    pseudo_config = json.loads(
        (args.boxes_root / "run_config.json").read_text(encoding="utf-8")
    )
    protocol = {
        "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": runtime.checkpoint_epoch,
        "weights": str(args.weights.resolve()),
        "boxes_root": str(args.boxes_root.resolve()),
        "bbox_source": "yolo26x_bytetrack_label_blind_primary_actor",
        "pseudo_label_config": pseudo_config,
        "feature_contract": runtime.contract,
        "video_score": "maximum window score per video",
        "threshold": float(args.threshold),
        "limit_per_class": int(args.limit_per_class),
        "locked_external_test": args.limit_per_class == 0,
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
    started = time.perf_counter()
    try:
        with (output_dir / "predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(VideoOutcome.__annotations__))
            writer.writeheader()
            for index, record in enumerate(records, start=1):
                boxes_path = _artifact_path(
                    args.boxes_root / "actor_boxes", record.sample_id, ".jsonl"
                )
                try:
                    outcome = _evaluate_record(
                        record,
                        boxes_root=args.boxes_root,
                        runtime=runtime,
                        feature_batch_size=args.feature_batch_size,
                        window_batch_size=args.window_batch_size,
                    )
                except Exception as error:
                    outcome = _empty_outcome(
                        record,
                        boxes_path,
                        status="error",
                        reason=f"{type(error).__name__}: {error}",
                    )
                outcomes.append(outcome)
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
    summary = {
        "protocol": protocol,
        "duration_s": time.perf_counter() - started,
        "metrics": _metrics(outcomes, threshold=args.threshold),
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
