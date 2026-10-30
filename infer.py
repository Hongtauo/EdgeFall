"""Run standalone HN3-B-R fall detection on one video or a directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from edgefall_st.runtime.fygr_online import OnlineFygrRuntime, run_video


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "hn3br_epoch008_inference.pt"
DEFAULT_YOLO = ROOT / "models" / "yolo26s.pt"
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _videos(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video extension: {path.suffix}")
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(path)
    iterator = path.rglob("*") if recursive else path.glob("*")
    videos = sorted(
        (item.resolve() for item in iterator if item.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda item: str(item).casefold(),
    )
    if not videos:
        raise ValueError(f"No supported videos found under {path}")
    return videos


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--threshold", type=float, default=0.7094)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--quantize", type=int, default=16)
    parser.add_argument("--max-actor-missed", type=int, default=2)
    parser.add_argument("--max-people", type=int, default=2)
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate(args: argparse.Namespace) -> None:
    for path in (args.checkpoint, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if min(args.max_people, args.max_actor_missed + 1, args.warmup_samples + 1) <= 0:
        raise ValueError("Invalid people, missed-track, or warm-up setting")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate(args)
    videos = _videos(args.input, args.recursive)
    output_dir = args.output_dir or (
        ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runtime = OnlineFygrRuntime(
        checkpoint_path=args.checkpoint.resolve(),
        weights_path=args.weights.resolve(),
        device=device,
        amp=args.amp,
        confidence=args.confidence,
        iou=args.iou,
        tracker=args.tracker,
        quantize=None if args.quantize <= 0 else args.quantize,
        max_actor_missed=args.max_actor_missed,
        max_people=args.max_people,
    )
    video_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    timed_algorithm_ms: list[float] = []
    try:
        for index, video_path in enumerate(videos):
            result, predictions = run_video(
                runtime,
                video_path,
                path=str(video_path),
                label=None,
                alarm_threshold=args.threshold,
                warmup_samples=args.warmup_samples if index == 0 else 0,
            )
            video_rows.append(
                {
                    **asdict(result),
                    "is_fall": int(result.video_score >= args.threshold),
                }
            )
            for prediction in predictions:
                row = asdict(prediction)
                row["path"] = str(video_path)
                row["actor_box"] = json.dumps(row["actor_box"])
                row["people"] = json.dumps(row["people"], ensure_ascii=False)
                frame_rows.append(row)
                if prediction.timed:
                    timed_algorithm_ms.append(prediction.algorithm_ms)
            print(
                f"[{index + 1}/{len(videos)}] "
                f"fall={int(result.video_score >= args.threshold)} "
                f"score={result.video_score:.4f} {video_path}",
                flush=True,
            )
    finally:
        runtime.close()

    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(video_rows[0]))
        writer.writeheader()
        writer.writerows(video_rows)
    with (output_dir / "frames.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    mean_ms = statistics.fmean(timed_algorithm_ms) if timed_algorithm_ms else 0.0
    gpu: dict[str, Any] | None = None
    peak_cuda_memory_mb = 0.0
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_gb": properties.total_memory / (1024**3),
            "torch_cuda_runtime": torch.version.cuda,
            "torch_cuda_arch_list": torch.cuda.get_arch_list(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "amp_dtype": str(runtime.amp_dtype).removeprefix("torch."),
        }
        peak_cuda_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    summary = {
        "model": "EdgeFall-ST HN3-B-R Seed-1 epoch 8",
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "videos": len(video_rows),
        "positive_predictions": sum(row["is_fall"] for row in video_rows),
        "alarm_threshold": args.threshold,
        "device": str(device),
        "gpu": gpu,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "yolo_weights": str(args.weights.resolve()),
        "yolo_sha256": _sha256(args.weights),
        "parameters": {
            "detector": runtime.detector_parameter_count,
            "confirmer_including_frozen_geometry": runtime.confirmer_parameter_count,
            "total": runtime.detector_parameter_count + runtime.confirmer_parameter_count,
        },
        "latency": {
            "timed_samples": len(timed_algorithm_ms),
            "algorithm_mean_ms": mean_ms,
            "algorithm_p95_ms": _percentile(timed_algorithm_ms, 95),
            "algorithm_fps": 1000.0 / mean_ms if mean_ms else 0.0,
            "peak_cuda_memory_mb": peak_cuda_memory_mb,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
