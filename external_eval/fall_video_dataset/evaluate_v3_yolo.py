"""Evaluate the frozen V3 checkpoint with generated YOLO primary-actor boxes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

from external_eval.fall_video_dataset.common import VideoRecord, discover_records
from external_eval.fall_video_dataset.evaluate_v3 import (
    PREDICTION_FIELDS,
    WindowPredictionObserver,
    NormalizedBoxSequence,
    VideoOutcome,
    _empty_outcome,
    _evaluate_video_with_box_loader,
    _load_model,
    _metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/fall-video-dataset"),
    )
    parser.add_argument(
        "--boxes-root",
        type=Path,
        default=Path(
            "artifacts/external_pseudo_labels/fall_video_dataset/yolo26x_nms"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/external_evaluation/fall_video_dataset"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-batch-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def _artifact_path(root: Path, sample_id: str, suffix: str) -> Path:
    relative = Path(*sample_id.split("/"))
    return root / relative.parent / f"{relative.name}{suffix}"


def read_actor_box_sequence(
    path: Path,
    *,
    frame_count: int,
) -> NormalizedBoxSequence:
    boxes: dict[int, tuple[float, float, float, float]] = {}
    observed_frames = 0
    maximum_frame = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            if frame_index < 0:
                raise ValueError(f"Negative frame index at {path}:{line_number}")
            observed_frames += 1
            maximum_frame = max(maximum_frame, frame_index)
            values = row.get("actor_bbox_xyxy_norm")
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"Invalid actor box at {path}:{line_number}")
            raw_box = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in raw_box):
                raise ValueError(f"Non-finite actor box at {path}:{line_number}")
            box = tuple(min(1.0, max(0.0, value)) for value in raw_box)
            left, top, right, bottom = box
            if left >= right or top >= bottom:
                continue
            boxes[frame_index] = box
    return NormalizedBoxSequence(
        boxes=boxes,
        source_path=path,
        observed_frames=observed_frames,
        valid_frames=len(boxes),
        frame_alignment_delta=(maximum_frame + 1) - frame_count,
        no_valid_reason="no_valid_yolo_actor_boxes",
    )


def _manifest_records(dataset_root: Path, boxes_root: Path) -> list[VideoRecord]:
    manifest_path = boxes_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    available = {
        record.sample_id: record for record in discover_records(dataset_root)
    }
    selected: list[VideoRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"Manifest has no sample_id column: {manifest_path}")
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id not in available:
                raise ValueError(f"Unknown sample in manifest: {sample_id}")
            selected.append(available[sample_id])
    if not selected:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return selected


def _evaluate_record(
    record: VideoRecord,
    *,
    boxes_root: Path,
    model,
    history_offsets: tuple[int, ...],
    effective_fps: float,
    crop_batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    window_observer: WindowPredictionObserver | None = None,
) -> VideoOutcome:
    boxes_path = _artifact_path(
        boxes_root / "actor_boxes", record.sample_id, ".jsonl"
    )

    def load_boxes(
        _width: int,
        _height: int,
        frame_count: int,
    ) -> NormalizedBoxSequence:
        return read_actor_box_sequence(boxes_path, frame_count=frame_count)

    return _evaluate_video_with_box_loader(
        record,
        model=model,
        history_offsets=history_offsets,
        effective_fps=effective_fps,
        box_loader=load_boxes,
        crop_batch_size=crop_batch_size,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        window_observer=window_observer,
    )


def evaluate(args: argparse.Namespace) -> Path:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.boxes_root.is_dir():
        raise FileNotFoundError(args.boxes_root)
    if args.crop_batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("batch and logging settings must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    records = _manifest_records(args.dataset_root, args.boxes_root)
    model, saved, offsets, amp_enabled, amp_dtype = _load_model(
        args.checkpoint,
        device,
    )
    run_name = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_yolo_boxes"
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.csv"
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    pseudo_config = json.loads(
        (args.boxes_root / "run_config.json").read_text(encoding="utf-8")
    )
    config = {
        "dataset_root": str(args.dataset_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "boxes_root": str(args.boxes_root.resolve()),
        "bbox_source": "yolo26x_bytetrack_label_blind_primary_actor",
        "pseudo_label_config": pseudo_config,
        "device": str(device),
        "model_parameter_count": model.parameter_count,
        "history_offsets": list(offsets),
        "effective_fps": float(saved.effective_fps),
        "crop_size": list(model.crop_size),
        "crop_context": float(model.cropper.context_scale),
        "video_score": "maximum sigmoid(fall_logits) across valid windows",
        "threshold": args.threshold,
        "evaluated_videos": len(records),
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
            boxes_path = _artifact_path(
                args.boxes_root / "actor_boxes", record.sample_id, ".jsonl"
            )
            try:
                outcome = _evaluate_record(
                    record,
                    boxes_root=args.boxes_root,
                    model=model,
                    history_offsets=offsets,
                    effective_fps=float(saved.effective_fps),
                    crop_batch_size=args.crop_batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            except Exception as error:
                outcome = _empty_outcome(
                    record,
                    status="error",
                    reason=f"{type(error).__name__}: {error}",
                    boxes_path=boxes_path,
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
