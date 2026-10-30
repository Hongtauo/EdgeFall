"""Evaluate frozen V3 and V3.1 with full YOLO boxes on Fall Video Dataset."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from edgefall_st.runtime.v31_events import (
    V31FallStateMachine,
    V31StateMachineConfig,
    V31WindowPrediction,
)
from external_eval.fall_video_dataset.evaluate_v3 import (
    PREDICTION_FIELDS,
    ExternalWindowPrediction,
    VideoOutcome,
    _empty_outcome,
    _load_model,
    _metrics,
)
from external_eval.fall_video_dataset.evaluate_v3_yolo import (
    _artifact_path,
    _evaluate_record,
    _manifest_records,
)


V31_PREDICTION_FIELDS = PREDICTION_FIELDS + (
    "v31_score",
    "v31_alarms",
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
    parser.add_argument("--confirmation-windows", type=int, default=2)
    parser.add_argument("--maximum-gap-s", type=float, default=0.6)
    parser.add_argument("--transition-window-s", type=float, default=2.0)
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--recovery-windows", type=int, default=3)
    parser.add_argument("--no-stage-transition", action="store_true")
    return parser


def _state_machine_result(
    predictions: Sequence[ExternalWindowPrediction],
    config: V31StateMachineConfig,
) -> tuple[float, int, list[dict[str, Any]]]:
    machine = V31FallStateMachine(config)
    score = 0.0
    alarm_count = 0
    events: list[dict[str, Any]] = []
    for prediction in predictions:
        update = machine.update(
            V31WindowPrediction(
                timestamp_s=prediction.timestamp_s,
                fall_score=prediction.fall_score,
                stage_probabilities=prediction.stage_probabilities,
                onset_score=prediction.onset_score,
            )
        )
        score = max(score, update.confirmation_score)
        for message in update.messages:
            if message.kind == "alarm":
                alarm_count += 1
            else:
                events.append(asdict(message))
    events.extend(asdict(message) for message in machine.flush())
    return score, alarm_count, events


def _fixed_threshold_effect(
    raw_outcomes: Sequence[VideoOutcome],
    v31_outcomes: Sequence[VideoOutcome],
    threshold: float,
) -> dict[str, int]:
    suppressed_positive = suppressed_negative = 0
    retained_positive = retained_negative = 0
    for raw, v31 in zip(raw_outcomes, v31_outcomes, strict=True):
        if raw.score is None or v31.score is None:
            continue
        raw_positive = raw.score >= threshold
        v31_positive = v31.score >= threshold
        if raw_positive and not v31_positive:
            if raw.label:
                suppressed_positive += 1
            else:
                suppressed_negative += 1
        elif raw_positive and v31_positive:
            if raw.label:
                retained_positive += 1
            else:
                retained_negative += 1
    return {
        "suppressed_positive_videos": suppressed_positive,
        "suppressed_negative_videos": suppressed_negative,
        "retained_positive_videos": retained_positive,
        "retained_negative_videos": retained_negative,
    }


def evaluate(args: argparse.Namespace) -> Path:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.boxes_root.is_dir():
        raise FileNotFoundError(args.boxes_root)
    if args.crop_batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("batch and logging settings must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    state_config = V31StateMachineConfig(
        confirmation_windows=args.confirmation_windows,
        maximum_gap_s=args.maximum_gap_s,
        transition_window_s=args.transition_window_s,
        alarm_threshold=args.threshold,
        release_threshold=args.release_threshold,
        recovery_windows=args.recovery_windows,
        use_stage_transition=not args.no_stage_transition,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    records = _manifest_records(args.dataset_root, args.boxes_root)
    model, saved, offsets, amp_enabled, amp_dtype = _load_model(
        args.checkpoint,
        device,
    )
    run_name = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_v31_yolo_full"
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.csv"
    window_predictions_path = output_dir / "window_predictions.jsonl"
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
        "raw_video_score": "maximum sigmoid(fall_logits)",
        "v31_video_score": (
            "maximum causal persistence or ordered stage confirmation"
        ),
        "state_machine": asdict(state_config),
        "evaluated_videos": len(records),
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    raw_outcomes: list[VideoOutcome] = []
    v31_outcomes: list[VideoOutcome] = []
    started = time.perf_counter()
    with (
        predictions_path.open("w", encoding="utf-8-sig", newline="") as handle,
        window_predictions_path.open("w", encoding="utf-8") as window_handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=V31_PREDICTION_FIELDS)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            boxes_path = _artifact_path(
                args.boxes_root / "actor_boxes",
                record.sample_id,
                ".jsonl",
            )
            windows: list[ExternalWindowPrediction] = []
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
                    window_observer=windows.extend,
                )
            except Exception as error:
                outcome = _empty_outcome(
                    record,
                    status="error",
                    reason=f"{type(error).__name__}: {error}",
                    boxes_path=boxes_path,
                )

            if outcome.score is None:
                v31_score = None
                alarm_count = 0
                events: list[dict[str, Any]] = []
            else:
                v31_score, alarm_count, events = _state_machine_result(
                    windows,
                    state_config,
                )
            raw_outcomes.append(outcome)
            v31_outcomes.append(replace(outcome, score=v31_score))
            row = asdict(outcome)
            row["v31_score"] = v31_score
            row["v31_alarms"] = alarm_count
            writer.writerow(row)
            handle.flush()

            for window in windows:
                window_handle.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "label": record.label,
                            **asdict(window),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if events:
                window_handle.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "label": record.label,
                            "events_at_fixed_threshold": events,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            window_handle.flush()

            if index % args.log_interval == 0 or index == len(records):
                elapsed = time.perf_counter() - started
                ok_count = sum(item.status == "ok" for item in raw_outcomes)
                print(
                    f"videos={index}/{len(records)} ok={ok_count} "
                    f"videos_per_s={index / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )

    summary = {
        "protocol": config,
        "duration_s": time.perf_counter() - started,
        "metrics": {
            "v3_raw_max": _metrics(raw_outcomes, threshold=args.threshold),
            "v31_state_machine": _metrics(
                v31_outcomes,
                threshold=args.threshold,
            ),
            "fixed_threshold_effect": _fixed_threshold_effect(
                raw_outcomes,
                v31_outcomes,
                args.threshold,
            ),
        },
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
