"""Detailed error analysis for bbox-conditioned fall checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from edgefall_st.data.labels import ACTION_LABELS, STAGE_LABELS
from edgefall_st.evaluation.checkpoint import (
    _make_dataset,
    _make_loader,
    _positive_video_paths,
    _resolve_device,
    _restore_saved_argument_types,
    _split_paths,
)
from edgefall_st.training.train import (
    _MetricAccumulator,
    _make_model_and_criterion,
    _move_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--tracks-root", type=Path)
    parser.add_argument("--actor-targets-root", type=Path)
    parser.add_argument("--actor-crops-root", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-examples", type=int, default=12)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers is not None and args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.max_batches < 0:
        raise ValueError("max-batches cannot be negative")
    if args.log_interval <= 0 or args.top_examples <= 0:
        raise ValueError("log-interval and top-examples must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> dict[str, float | None]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positive_count = sum(int(label) for label in labels)
    if not scores or positive_count == 0:
        return {
            "average_precision": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "mean_precision": 0.0,
            "p90_threshold": None,
            "p95_threshold": None,
        }
    records = sorted(
        zip(scores, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    curve: list[dict[str, float]] = []
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(records):
        threshold = float(records[index][0])
        group_true_positives = 0
        group_size = 0
        while index < len(records) and records[index][0] == threshold:
            group_true_positives += int(records[index][1])
            group_size += 1
            index += 1
        true_positives += group_true_positives
        false_positives += group_size - group_true_positives
        recall = true_positives / positive_count
        precision = true_positives / max(true_positives + false_positives, 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        curve.append(
            {
                "threshold": threshold,
                "recall": recall,
                "precision": precision,
            }
        )

    def at_recall(target: float) -> tuple[float, float | None]:
        candidates = [point for point in curve if point["recall"] >= target]
        if not candidates:
            return 0.0, None
        best = max(
            candidates,
            key=lambda point: (point["precision"], point["threshold"]),
        )
        return best["precision"], best["threshold"]

    p90, threshold90 = at_recall(0.90)
    p95, threshold95 = at_recall(0.95)
    return {
        "average_precision": average_precision,
        "p90": p90,
        "p95": p95,
        "mean_precision": (p90 + p95) * 0.5,
        "p90_threshold": threshold90,
        "p95_threshold": threshold95,
    }


def _classification_counts(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, float | int]:
    true_positives = sum(
        int(record["fall_target"] and record["fall_score"] >= threshold)
        for record in records
    )
    false_positives = sum(
        int(not record["fall_target"] and record["fall_score"] >= threshold)
        for record in records
    )
    false_negatives = sum(
        int(record["fall_target"] and record["fall_score"] < threshold)
        for record in records
    )
    true_negatives = len(records) - true_positives - false_positives - false_negatives
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2.0 * precision * recall, precision + recall),
    }


def _per_action(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["action_label"])].append(record)
    result: dict[str, Any] = {}
    for action_id in sorted(ACTION_LABELS):
        values = grouped.get(action_id, [])
        positive = action_id in (1, 2)
        predicted_positive = sum(
            int(record["fall_score"] >= threshold) for record in values
        )
        stage_predictions = Counter(
            STAGE_LABELS[int(record["stage_prediction"])] for record in values
        )
        row: dict[str, Any] = {
            "action_id": action_id,
            "support": len(values),
            "fall_target": positive,
            "predicted_positive": predicted_positive,
            "predicted_positive_rate": _safe_divide(
                predicted_positive,
                len(values),
            ),
            "fall_score": _quantiles(
                [float(record["fall_score"]) for record in values]
            ),
            "stage_predictions": dict(stage_predictions),
            "stage_accuracy": _safe_divide(
                sum(
                    int(record["stage_prediction"] == record["stage_target"])
                    for record in values
                ),
                len(values),
            ),
        }
        row["recall" if positive else "false_positive_rate"] = (
            _safe_divide(predicted_positive, len(values))
        )
        result[ACTION_LABELS[action_id]] = row
    return result


def _stage_confusion(records: Sequence[dict[str, Any]]) -> list[list[int]]:
    confusion = [[0 for _ in range(3)] for _ in range(3)]
    for record in records:
        confusion[int(record["stage_target"])][
            int(record["stage_prediction"])
        ] += 1
    return confusion


def _onset_summary(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    true_positives = sum(
        int(record["onset_target"] and record["onset_score"] >= threshold)
        for record in records
    )
    false_positives = sum(
        int(not record["onset_target"] and record["onset_score"] >= threshold)
        for record in records
    )
    false_negatives = sum(
        int(record["onset_target"] and record["onset_score"] < threshold)
        for record in records
    )
    false_positive_actions = Counter(
        record["action_name"]
        for record in records
        if not record["onset_target"] and record["onset_score"] >= threshold
    )
    return {
        "threshold": threshold,
        "positive_support": sum(int(record["onset_target"]) for record in records),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": _safe_divide(true_positives, true_positives + false_positives),
        "recall": _safe_divide(true_positives, true_positives + false_negatives),
        "positive_scores": _quantiles(
            [record["onset_score"] for record in records if record["onset_target"]]
        ),
        "negative_scores": _quantiles(
            [record["onset_score"] for record in records if not record["onset_target"]]
        ),
        "false_positive_actions": dict(false_positive_actions.most_common()),
    }


def _strata(
    records: Sequence[dict[str, Any]],
    threshold: float,
    key: str,
    bins: Sequence[tuple[str, float, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, lower, upper in bins:
        values = [
            record
            for record in records
            if lower <= float(record[key]) < upper
        ]
        positive_values = [record for record in values if record["fall_target"]]
        negative_values = [record for record in values if not record["fall_target"]]
        result[name] = {
            "support": len(values),
            "positive_support": len(positive_values),
            "negative_support": len(negative_values),
            "positive_recall": _safe_divide(
                sum(
                    int(record["fall_score"] >= threshold)
                    for record in positive_values
                ),
                len(positive_values),
            ),
            "negative_false_positive_rate": _safe_divide(
                sum(
                    int(record["fall_score"] >= threshold)
                    for record in negative_values
                ),
                len(negative_values),
            ),
        }
    return result


def _error_action_counts(
    records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    false_positive: bool,
) -> dict[str, int]:
    counts = Counter()
    for record in records:
        predicted = record["fall_score"] >= threshold
        selected = (
            predicted and not record["fall_target"]
            if false_positive
            else not predicted and record["fall_target"]
        )
        if selected:
            counts[str(record["action_name"])] += 1
    return dict(counts.most_common())


def _video_analysis(
    records: Sequence[dict[str, Any]],
    positive_paths: set[str] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["path"])].append(record)
    videos: list[dict[str, Any]] = []
    for path, values in grouped.items():
        peak = max(values, key=lambda record: record["fall_score"])
        positives = [record for record in values if record["fall_target"]]
        video_target = bool(positives) if positive_paths is None else path in positive_paths
        positive_peak = (
            max(positives, key=lambda record: record["fall_score"])
            if positives
            else None
        )
        videos.append(
            {
                "path": path,
                "fall_target": video_target,
                "fall_score": float(peak["fall_score"]),
                "peak_record": peak,
                "positive_peak_record": positive_peak,
            }
        )
    ranking = _ranking_metrics(
        [video["fall_score"] for video in videos],
        [video["fall_target"] for video in videos],
    )
    result: dict[str, Any] = {
        "videos": len(videos),
        "positive_videos": sum(int(video["fall_target"]) for video in videos),
        **ranking,
    }
    for label in ("p90", "p95"):
        threshold_value = ranking[f"{label}_threshold"]
        if threshold_value is None:
            continue
        threshold = float(threshold_value)
        false_positives = [
            video
            for video in videos
            if not video["fall_target"] and video["fall_score"] >= threshold
        ]
        false_negatives = [
            video
            for video in videos
            if video["fall_target"] and video["fall_score"] < threshold
        ]
        result[f"{label}_errors"] = {
            "threshold": threshold,
            "false_positive_videos": len(false_positives),
            "false_negative_videos": len(false_negatives),
            "false_positive_culprit_actions": dict(
                Counter(
                    video["peak_record"]["action_name"]
                    for video in false_positives
                ).most_common()
            ),
            "false_positives": sorted(
                false_positives,
                key=lambda video: video["fall_score"],
                reverse=True,
            ),
            "false_negatives": sorted(
                false_negatives,
                key=lambda video: video["fall_score"],
            ),
        }
    return result


def _top_unique_paths(
    records: Iterable[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        path = str(record["path"])
        if path in seen:
            continue
        seen.add(path)
        selected.append(record)
        if len(selected) >= count:
            break
    return selected


def _read_actor_box(actor_path: Path, frame_index: int) -> list[float] | None:
    with actor_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index != frame_index:
                continue
            row = json.loads(line)
            values = row.get("actor_bbox_xyxy_norm")
            if isinstance(values, list) and len(values) == 4:
                return [float(value) for value in values]
            return None
    return None


def _render_examples(
    records: Sequence[dict[str, Any]],
    *,
    output: Path,
    title: str,
    dataset_root: Path,
    actor_targets_root: Path,
    video_suffix: str,
) -> None:
    tile_w, image_h, text_h = 360, 203, 67
    tile_h = image_h + text_h
    columns = 3
    rows = max(1, math.ceil(len(records) / columns))
    canvas = np.full((rows * tile_h + 40, columns * tile_w, 3), 245, np.uint8)
    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for item_index, record in enumerate(records):
        row = item_index // columns
        column = item_index % columns
        x0 = column * tile_w
        y0 = 40 + row * tile_h
        relative = Path(str(record["path"]).replace("\\", "/"))
        video_path = (dataset_root / relative).with_suffix(video_suffix)
        actor_path = (actor_targets_root / relative).with_suffix(".jsonl")
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["source_frame_index"]))
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            frame = np.full((image_h, tile_w, 3), 90, np.uint8)
        source_h, source_w = frame.shape[:2]
        box = _read_actor_box(actor_path, int(record["source_frame_index"]))
        if box is not None:
            left = int(box[0] * source_w)
            top = int(box[1] * source_h)
            right = int(box[2] * source_w)
            bottom = int(box[3] * source_h)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 3)
        resized = cv2.resize(frame, (tile_w, image_h), interpolation=cv2.INTER_AREA)
        canvas[y0 : y0 + image_h, x0 : x0 + tile_w] = resized
        lines = (
            f"{record['action_name']} score={record['fall_score']:.3f}",
            f"stage={record['stage_target']}->{record['stage_prediction']} "
            f"onset={record['onset_score']:.3f}",
            f"{record['path']} t={record['end_time_s']:.2f}s",
        )
        for line_index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line[:52],
                (x0 + 5, y0 + image_h + 19 + line_index * 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write visualization: {output}")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**checkpoint["arguments"])
    _restore_saved_argument_types(saved)
    if getattr(saved, "training_mode", "unified") != "bbox-classifier":
        raise ValueError("This analyzer requires a bbox-classifier checkpoint")
    (
        dataset_root,
        windows,
        tracks_root,
        actor_targets_root,
        actor_crops_root,
        events,
    ) = _split_paths(args, saved)
    dataset = _make_dataset(
        saved,
        dataset_root=dataset_root,
        windows=windows,
        tracks_root=tracks_root,
        actor_targets_root=actor_targets_root,
        actor_crops_root=actor_crops_root,
    )
    batch_size = args.batch_size or int(saved.batch_size)
    workers = int(saved.workers) if args.workers is None else int(args.workers)
    loader = _make_loader(dataset, saved, batch_size=batch_size, workers=workers)
    model, criterion = _make_model_and_criterion(saved)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    criterion.to(device).eval()
    amp_enabled = bool(saved.amp) and device.type == "cuda"
    amp_dtype = torch.float16 if saved.amp_dtype == "float16" else torch.bfloat16
    if amp_dtype == torch.bfloat16 and device.type == "cuda":
        amp_enabled = amp_enabled and torch.cuda.is_bf16_supported()

    maximum_batches = (
        min(args.max_batches, len(loader))
        if args.max_batches > 0
        else len(loader)
    )
    records: list[dict[str, Any]] = []
    losses = _MetricAccumulator()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            images = batch["images"].to(device, non_blocking=saved.pin_memory)
            images = images.to(torch.float32).div_(255.0)
            actor_boxes = batch["actor_boxes"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            actor_valid = batch["actor_valid_mask"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            history_indices = batch["history_indices"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            valid_mask = batch["valid_mask"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            targets = _move_targets(batch["targets"], device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model.forward_indexed_history(
                    images,
                    actor_boxes,
                    history_indices,
                    valid_mask,
                    actor_valid,
                )
                batch_losses = criterion(outputs, targets, batch["metadata"])
            losses.update(batch_losses, len(targets))

            stage_probabilities = outputs["stage_logits"].float().softmax(-1).cpu()
            fall_scores = outputs["fall_logits"].float().sigmoid().cpu()
            onset_scores = outputs["onset_logits"].float().sigmoid().cpu()
            cpu_indices = history_indices.cpu()
            cpu_actor_boxes = actor_boxes.float().cpu()
            cpu_actor_valid = actor_valid.float().cpu()
            cpu_valid_mask = valid_mask.float().cpu()
            for sample_index, (target, metadata) in enumerate(
                zip(targets, batch["metadata"], strict=True)
            ):
                if float(target["action_loss_weight"].detach().cpu()) <= 0:
                    continue
                safe_indices = cpu_indices[sample_index].clamp_min(0)
                history_actor_valid = cpu_actor_valid[safe_indices]
                effective_valid = (
                    cpu_valid_mask[sample_index]
                    * cpu_indices[sample_index].ge(0)
                    * history_actor_valid
                )
                current_index = int(cpu_indices[sample_index, 0])
                current_box = cpu_actor_boxes[current_index]
                width = float((current_box[2] - current_box[0]).clamp_min(0))
                height = float((current_box[3] - current_box[1]).clamp_min(1e-6))
                action_label = int(target["action_label"].detach().cpu())
                stage_target = int(target["stage_label"].detach().cpu())
                probabilities = stage_probabilities[sample_index]
                records.append(
                    {
                        "path": str(metadata["path"]),
                        "window_index": int(metadata["window_index"]),
                        "end_time_s": float(metadata["end_time_s"]),
                        "source_frame_index": int(metadata["source_frame_index"]),
                        "action_label": action_label,
                        "action_name": ACTION_LABELS[action_label],
                        "stage_target": stage_target,
                        "stage_target_name": STAGE_LABELS[stage_target],
                        "stage_prediction": int(probabilities.argmax()),
                        "stage_probabilities": [
                            float(value) for value in probabilities.tolist()
                        ],
                        "fall_target": bool(
                            metadata.get(
                                "event_target",
                                stage_target != 0,
                            )
                        ),
                        "fall_score": float(fall_scores[sample_index]),
                        "onset_target": bool(
                            float(target["onset_target"].detach().cpu()) >= 0.5
                        ),
                        "onset_score": float(onset_scores[sample_index]),
                        "bbox": [float(value) for value in current_box.tolist()],
                        "bbox_area": width * height,
                        "bbox_aspect": width / height,
                        "history_valid_count": int(effective_valid.sum()),
                    }
                )
            completed = batch_index + 1
            if completed % args.log_interval == 0 or completed == maximum_batches:
                print(
                    f"batch={completed}/{maximum_batches} records={len(records)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )

    ranking = _ranking_metrics(
        [record["fall_score"] for record in records],
        [record["fall_target"] for record in records],
    )
    all_video_paths = {str(record["path"]) for record in records}
    event_positive_paths = _positive_video_paths(events, all_video_paths)
    video_from_stages = _video_analysis(records)
    video = _video_analysis(records, event_positive_paths)
    threshold = float(args.threshold)
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "split": args.split,
        "duration_s": time.perf_counter() - started,
        "windows": len(records),
        "losses": losses.result(),
        "window": {
            **ranking,
            "at_default_threshold": _classification_counts(records, threshold),
            "p90_false_positive_actions": _error_action_counts(
                records,
                threshold=float(ranking["p90_threshold"]),
                false_positive=True,
            ),
            "p90_false_negative_actions": _error_action_counts(
                records,
                threshold=float(ranking["p90_threshold"]),
                false_positive=False,
            ),
            "p95_false_positive_actions": _error_action_counts(
                records,
                threshold=float(ranking["p95_threshold"]),
                false_positive=True,
            ),
            "p95_false_negative_actions": _error_action_counts(
                records,
                threshold=float(ranking["p95_threshold"]),
                false_positive=False,
            ),
        },
        "video": video,
        "video_from_window_stages": video_from_stages,
        "video_target_definition": {
            "primary": "path appears in events.csv",
            "event_positive_videos": len(event_positive_paths),
            "stage_positive_videos": video_from_stages["positive_videos"],
        },
        "per_action": _per_action(records, threshold),
        "stage_confusion": _stage_confusion(records),
        "onset": _onset_summary(records, threshold),
        "strata": {
            "bbox_area": _strata(
                records,
                threshold,
                "bbox_area",
                (
                    ("tiny", 0.0, 0.05),
                    ("small", 0.05, 0.15),
                    ("medium", 0.15, 0.30),
                    ("large", 0.30, math.inf),
                ),
            ),
            "bbox_aspect": _strata(
                records,
                threshold,
                "bbox_aspect",
                (
                    ("vertical", 0.0, 0.75),
                    ("square", 0.75, 1.25),
                    ("horizontal", 1.25, math.inf),
                ),
            ),
            "history_valid_count": _strata(
                records,
                threshold,
                "history_valid_count",
                (
                    ("cold_start", 0.0, 4.0),
                    ("partial", 4.0, 8.0),
                    ("full", 8.0, 9.0),
                ),
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    window_false_positives = _top_unique_paths(
        sorted(
            (
                record
                for record in records
                if not record["fall_target"] and record["fall_score"] >= threshold
            ),
            key=lambda record: record["fall_score"],
            reverse=True,
        ),
        args.top_examples,
    )
    window_false_negatives = _top_unique_paths(
        sorted(
            (
                record
                for record in records
                if record["fall_target"] and record["fall_score"] < threshold
            ),
            key=lambda record: record["fall_score"],
        ),
        args.top_examples,
    )
    p95_errors = video.get("p95_errors", {})
    video_false_positives = [
        item["peak_record"]
        for item in p95_errors.get("false_positives", [])[: args.top_examples]
    ]
    video_false_negatives = [
        item["positive_peak_record"] or item["peak_record"]
        for item in p95_errors.get("false_negatives", [])[: args.top_examples]
    ]
    visualizations = (
        ("window_false_positives.jpg", "Window FP at threshold 0.5", window_false_positives),
        ("window_false_negatives.jpg", "Window FN at threshold 0.5", window_false_negatives),
        ("video_p95_false_positives.jpg", "Video FP at P95 threshold", video_false_positives),
        ("video_p95_false_negatives.jpg", "Video FN at P95 threshold", video_false_negatives),
    )
    for filename, title, examples in visualizations:
        _render_examples(
            examples,
            output=args.output_dir / filename,
            title=title,
            dataset_root=dataset_root,
            actor_targets_root=actor_targets_root,
            video_suffix=saved.video_suffix,
        )
    summary["example_records"] = {
        "window_false_positives": window_false_positives,
        "window_false_negatives": window_false_negatives,
        "video_p95_false_positives": video_false_positives,
        "video_p95_false_negatives": video_false_negatives,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path.resolve())}, indent=2))
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    analyze(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
