"""Cache-free online evaluation on UR Fall Detection RGB frame sequences."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch

from edgefall_st.runtime.fygr_online import (
    OnlineFygrRuntime,
    OnlinePrediction,
    VideoOnlineResult,
    _binary_metrics,
    _device,
    _percentile,
    _ranking_metrics,
    _sampled_source_indices,
)
from external_eval.ur_fall_detection_dataset.common import (
    UrFallSequence,
    discover_sequences,
)


DEFAULT_THRESHOLD = 0.6027905941009521


def run_sequence(
    runtime: OnlineFygrRuntime,
    sequence: UrFallSequence,
    *,
    source_fps: float,
    alarm_threshold: float,
    warmup_samples: int,
) -> tuple[VideoOnlineResult, list[OnlinePrediction]]:
    source_indices = _sampled_source_indices(
        len(sequence.frame_paths), source_fps, runtime.effective_fps
    )
    runtime.reset_stream()
    predictions: list[OnlinePrediction] = []
    decode_ms = 0.0
    wall_started = time.perf_counter()
    for sample_index, source_index in enumerate(source_indices):
        decode_started = time.perf_counter()
        frame = cv2.imread(str(sequence.frame_paths[source_index]))
        decode_ms += (time.perf_counter() - decode_started) * 1000.0
        if frame is None:
            raise ValueError(
                f"Could not decode frame: {sequence.frame_paths[source_index]}"
            )
        predictions.append(
            runtime.process_frame(frame, sample_index=sample_index)
        )
    wall_s = time.perf_counter() - wall_started
    scores = [prediction.score for prediction in predictions]
    first_alarm = next(
        (
            prediction.timestamp_s
            for prediction in predictions
            if prediction.score >= alarm_threshold
        ),
        None,
    )
    if warmup_samples > 0:
        predictions[:warmup_samples] = [
            replace(prediction, timed=False)
            for prediction in predictions[:warmup_samples]
        ]
    return (
        VideoOnlineResult(
            path=sequence.sample_id,
            label=int(sequence.label),
            source_fps=float(source_fps),
            source_frames=len(sequence.frame_paths),
            sampled_frames=len(predictions),
            video_score=max(scores, default=0.0),
            first_alarm_s=first_alarm,
            decode_ms=decode_ms,
            wall_s=wall_s,
        ),
        predictions,
    )


def _latency_report(
    predictions: Sequence[tuple[str, OnlinePrediction]],
    videos: Sequence[VideoOnlineResult],
    *,
    device: torch.device,
) -> dict[str, Any]:
    timed = [prediction for _, prediction in predictions if prediction.timed]
    latency: dict[str, Any] = {"timed_frames": len(timed)}
    for field in (
        "detector_ms",
        "tracking_ms",
        "feature_pool_ms",
        "confirmer_ms",
        "algorithm_ms",
    ):
        values = [float(getattr(prediction, field)) for prediction in timed]
        latency[field] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
        }
    algorithm_mean = latency["algorithm_ms"]["mean"]
    latency["algorithm_fps"] = (
        1000.0 / algorithm_mean if algorithm_mean else 0.0
    )
    latency["decode_ms_total"] = sum(video.decode_ms for video in videos)
    latency["wall_s_total"] = sum(video.wall_s for video in videos)
    latency["peak_cuda_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return latency


def _write_outputs(
    output_dir: Path,
    videos: Sequence[VideoOnlineResult],
    predictions: Sequence[tuple[str, OnlinePrediction]],
    report: dict[str, Any],
) -> None:
    with (output_dir / "videos.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(videos[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(video) for video in videos)
    with (output_dir / "frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["path", *asdict(predictions[0][1]).keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {"path": path, **asdict(prediction)}
            for path, prediction in predictions
        )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/ur-fall-detection-dataset"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/checkpoints/20260730_231605_455025/best.pt"
        ),
    )
    parser.add_argument("--weights", type=Path, default=Path("yolo26s.pt"))
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument(
        "--alarm-threshold", type=float, default=DEFAULT_THRESHOLD
    )
    parser.add_argument(
        "--threshold-source",
        default="V3.6-CAUCA60 pooled three-domain validation F1; URFD excluded",
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--quantize", type=int, default=16)
    parser.add_argument("--max-actor-missed", type=int, default=2)
    parser.add_argument("--max-people", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/external_evaluation/ur_fall_detection_dataset/"
            "v36_cauca60_yolo26s_online"
        ),
    )
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_fps <= 0 or not 0.0 <= args.alarm_threshold <= 1.0:
        raise ValueError("Invalid source FPS or alarm threshold")
    if min(args.max_people, args.log_interval, args.warmup_samples + 1) <= 0:
        raise ValueError("Invalid people, logging, or warm-up setting")
    records = discover_sequences(args.dataset_root)
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = OnlineFygrRuntime(
        checkpoint_path=args.checkpoint,
        weights_path=args.weights,
        device=device,
        amp=args.amp,
        confidence=args.confidence,
        iou=args.iou,
        tracker=args.tracker,
        quantize=args.quantize,
        max_actor_missed=args.max_actor_missed,
        max_people=args.max_people,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    videos: list[VideoOnlineResult] = []
    predictions: list[tuple[str, OnlinePrediction]] = []
    try:
        for index, sequence in enumerate(records, start=1):
            video, frame_predictions = run_sequence(
                runtime,
                sequence,
                source_fps=args.source_fps,
                alarm_threshold=args.alarm_threshold,
                warmup_samples=args.warmup_samples if index == 1 else 0,
            )
            videos.append(video)
            predictions.extend(
                (video.path, prediction) for prediction in frame_predictions
            )
            if index % args.log_interval == 0 or index == len(records):
                print(
                    f"sequences={index}/{len(records)} "
                    f"path={video.path} score={video.video_score:.4f} "
                    f"sampled={video.sampled_frames} wall_s={video.wall_s:.2f}",
                    flush=True,
                )
    finally:
        runtime.close()

    report = {
        "dataset": "UR Fall Detection Dataset cam0 RGB",
        "checkpoint": str(args.checkpoint),
        "weights": str(args.weights),
        "sequences": len(videos),
        "fall_sequences": sum(int(video.label == 1) for video in videos),
        "adl_sequences": sum(int(video.label == 0) for video in videos),
        "source_frames": sum(video.source_frames for video in videos),
        "sampled_frames": sum(video.sampled_frames for video in videos),
        "source_fps": args.source_fps,
        "effective_fps": runtime.effective_fps,
        "alarm_threshold": args.alarm_threshold,
        "threshold_source": args.threshold_source,
        "ranking": _ranking_metrics(videos),
        "fixed_threshold": _binary_metrics(videos, args.alarm_threshold),
        "latency": _latency_report(predictions, videos, device=device),
        "parameters": {
            "detector": runtime.detector_parameter_count,
            "detector_runtime_fused": runtime.fused_detector_parameter_count,
            "confirmer_including_frozen_geometry": runtime.confirmer_parameter_count,
            "total": (
                runtime.detector_parameter_count
                + runtime.confirmer_parameter_count
            ),
            "total_runtime_fused": (
                runtime.fused_detector_parameter_count
                + runtime.confirmer_parameter_count
            ),
        },
        "max_people": args.max_people,
        "causal_tracking": True,
        "disk_feature_cache": False,
        "video_score": "maximum causal sampled-frame fall score",
    }
    _write_outputs(args.output_dir, videos, predictions, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"output={args.output_dir}", flush=True)
    return report


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
