"""Evaluate a trained EdgeFall-ST checkpoint on fixed OmniFall windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms

from edgefall_st.data.dataset import (
    OmniFallWindowDataset,
    omnifall_indexed_collate_fn,
)
from edgefall_st.training.train import (
    _ClassifierMetricAccumulator,
    _MetricAccumulator,
    _classifier_history_offsets,
    _make_model_and_criterion,
    _move_targets,
)


def resolve_checkpoint_dependency(
    saved_path: str | Path,
    checkpoint_path: str | Path,
) -> Path:
    """Resolve saved dependencies from the working directory or checkpoint.

    Training checkpoints historically store repository-relative paths. Portable
    release checkpoints instead store sibling filenames. Prefer an existing
    working-directory path to preserve the original training behavior.
    """
    dependency = Path(saved_path)
    candidates = [dependency]
    if not dependency.is_absolute():
        candidates.append(Path(checkpoint_path).resolve().parent / dependency)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"Checkpoint dependency {str(saved_path)!r} was not found; "
        f"checkpoint={Path(checkpoint_path).resolve()}; searched: {searched}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--tracks-root", type=Path)
    parser.add_argument("--actor-targets-root", type=Path)
    parser.add_argument("--actor-crops-root", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--person-threshold", type=float, default=0.001)
    parser.add_argument("--report-person-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk", type=int, default=300)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers is not None and args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.max_batches < 0:
        raise ValueError("max-batches cannot be negative")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    for name in (
        "person_threshold",
        "report_person_threshold",
        "nms_iou_threshold",
        "match_iou_threshold",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")
    if args.pre_nms_topk <= 0 or args.max_detections <= 0:
        raise ValueError("pre-nms-topk and max-detections must be positive")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _restore_saved_argument_types(saved: argparse.Namespace) -> None:
    saved.regression_ranges = [
        (
            math.inf
            if isinstance(value, str)
            and value.strip().lower() in {".inf", "inf", "+inf", "+.inf"}
            else float(value)
        )
        for value in saved.regression_ranges
    ]


def _split_paths(
    args: argparse.Namespace,
    saved: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path | None, Path]:
    dataset_root = args.dataset_root or Path(saved.dataset_root)
    if args.split == "val":
        windows = args.windows or Path(saved.val_windows)
        tracks_root = args.tracks_root or Path(saved.val_tracks_root)
        actor_targets_root = (
            args.actor_targets_root or Path(saved.val_actor_targets_root)
        )
        saved_crop_root = getattr(saved, "val_actor_crops_root", None)
        actor_crops_root = args.actor_crops_root or (
            None if saved_crop_root is None else Path(saved_crop_root)
        )
    else:
        saved_val_windows = Path(saved.val_windows)
        windows = args.windows or saved_val_windows.with_name(
            f"windows_{args.split}.jsonl"
        )
        tracks_root = args.tracks_root or Path(
            f"artifacts/pseudo_labels/{args.split}_yolo26x_nms/tracks"
        )
        actor_targets_root = args.actor_targets_root or Path(
            f"artifacts/actor_targets_v1/{args.split}/targets"
        )
        training_used_crops = (
            getattr(saved, "train_actor_crops_root", None) is not None
        )
        actor_crops_root = args.actor_crops_root or (
            Path(f"artifacts/actor_crops_v1/{args.split}/crops")
            if training_used_crops
            else None
        )
    events = args.events or windows.parent / "events.csv"
    return (
        dataset_root,
        windows,
        tracks_root,
        actor_targets_root,
        actor_crops_root,
        events,
    )


def _history_offsets(saved: argparse.Namespace) -> tuple[int, ...]:
    if getattr(saved, "training_mode", None) == "bbox-classifier":
        return _classifier_history_offsets(saved)
    return tuple(
        sorted(
            {
                int(offset)
                for offsets in (
                    saved.history_p3,
                    saved.history_p4,
                    saved.history_p5,
                )
                for offset in offsets
            }
        )
    )


def _make_dataset(
    saved: argparse.Namespace,
    *,
    dataset_root: Path,
    windows: Path,
    tracks_root: Path,
    actor_targets_root: Path,
    actor_crops_root: Path | None,
) -> OmniFallWindowDataset:
    return OmniFallWindowDataset(
        dataset_root=dataset_root,
        windows_path=windows,
        tracks_root=tracks_root,
        actor_targets_root=actor_targets_root,
        actor_crops_root=actor_crops_root,
        actor_crop_size=(saved.crop_height, saved.crop_width),
        input_size=(saved.input_height, saved.input_width),
        effective_fps=saved.effective_fps,
        history_offsets=_history_offsets(saved),
        annotation_cache_size=saved.annotation_cache_size,
        video_frame_cache_size=saved.video_frame_cache_size,
        return_uint8=True,
        onset_duration_s=saved.onset_duration,
        video_suffix=saved.video_suffix,
        validate_files=saved.validate_files,
    )


def _make_loader(
    dataset: OmniFallWindowDataset,
    saved: argparse.Namespace,
    *,
    batch_size: int,
    workers: int,
) -> DataLoader[Any]:
    options: dict[str, Any] = {}
    if workers > 0:
        options["prefetch_factor"] = saved.prefetch_factor
        options["persistent_workers"] = saved.persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        collate_fn=omnifall_indexed_collate_fn,
        pin_memory=saved.pin_memory,
        **options,
    )


def _select_predictions(
    outputs: dict[str, Tensor],
    *,
    person_threshold: float,
    nms_iou_threshold: float,
    pre_nms_topk: int,
    max_detections: int,
) -> list[Tensor]:
    boxes = outputs["boxes"].float()
    scores = outputs["objectness_logits"].float().sigmoid()
    selected_batches: list[Tensor] = []
    for batch_index in range(boxes.shape[0]):
        indices = torch.nonzero(
            scores[batch_index] >= person_threshold,
            as_tuple=False,
        ).squeeze(1)
        if indices.numel() == 0:
            selected_batches.append(indices)
            continue
        if indices.numel() > pre_nms_topk:
            local_scores = scores[batch_index, indices]
            indices = indices[
                torch.topk(local_scores, pre_nms_topk, sorted=True).indices
            ]
        kept = nms(
            boxes[batch_index, indices],
            scores[batch_index, indices],
            nms_iou_threshold,
        )[:max_detections]
        selected_batches.append(indices[kept])
    return selected_batches


def _average_precision(
    records: Sequence[tuple[float, bool]],
    positive_count: int,
) -> float:
    if positive_count <= 0:
        return 0.0
    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    for _, is_true_positive in ordered:
        if is_true_positive:
            true_positives += 1
        else:
            false_positives += 1
        recall = true_positives / positive_count
        precision = true_positives / max(
            true_positives + false_positives,
            1,
        )
        if recall > previous_recall:
            average_precision += (recall - previous_recall) * precision
            previous_recall = recall
    return average_precision


def _precision_recall_at_threshold(
    records: Sequence[tuple[float, bool]],
    positive_count: int,
    threshold: float,
) -> dict[str, float | int]:
    selected = [record for record in records if record[0] >= threshold]
    true_positives = sum(int(record[1]) for record in selected)
    false_positives = len(selected) - true_positives
    false_negatives = max(0, positive_count - true_positives)
    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        ),
        "recall": (
            true_positives / positive_count if positive_count else 0.0
        ),
    }


def _binary_video_metrics(
    scores: dict[str, float],
    positive_paths: set[str],
) -> dict[str, Any]:
    ordered_scores = sorted(set(scores.values()), reverse=True)
    curve: list[dict[str, float | int]] = []
    for threshold in ordered_scores:
        selected = {
            path for path, score in scores.items() if score >= threshold
        }
        true_positives = len(selected & positive_paths)
        false_positives = len(selected - positive_paths)
        false_negatives = len(positive_paths - selected)
        curve.append(
            {
                "threshold": threshold,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": (
                    true_positives / (true_positives + false_positives)
                    if true_positives + false_positives
                    else 0.0
                ),
                "recall": (
                    true_positives / len(positive_paths)
                    if positive_paths
                    else 0.0
                ),
            }
        )

    def precision_at_recall(target: float) -> tuple[float, float | None]:
        candidates = [
            point for point in curve if float(point["recall"]) >= target
        ]
        if not candidates:
            return 0.0, None
        best = max(
            candidates,
            key=lambda point: (
                float(point["precision"]),
                float(point["threshold"]),
            ),
        )
        return float(best["precision"]), float(best["threshold"])

    records = [
        (score, path in positive_paths) for path, score in scores.items()
    ]
    p90, p90_threshold = precision_at_recall(0.90)
    p95, p95_threshold = precision_at_recall(0.95)
    return {
        "videos": len(scores),
        "positive_videos": len(positive_paths),
        "average_precision": _average_precision(
            records,
            len(positive_paths),
        ),
        "p90": p90,
        "p95": p95,
        "mean_precision": (p90 + p95) / 2.0,
        "p90_threshold": p90_threshold,
        "p95_threshold": p95_threshold,
    }


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


class _Evaluator:
    def __init__(
        self,
        *,
        match_iou_threshold: float,
        report_person_threshold: float,
    ) -> None:
        self.match_iou_threshold = match_iou_threshold
        self.report_person_threshold = report_person_threshold
        self.samples = 0
        self.loss_sums: defaultdict[str, float] = defaultdict(float)
        self.gt_boxes = 0
        self.detection_records: list[tuple[float, bool]] = []
        self.actor_windows = 0
        self.actor_detected = 0
        self.stage_correct_detected = 0
        self.action_correct_detected = 0
        self.stage_confusion = [[0 for _ in range(3)] for _ in range(3)]
        self.action_confusion = [[0 for _ in range(16)] for _ in range(16)]
        self.onset_tp = 0
        self.onset_fp = 0
        self.onset_fn = 0
        self.fall_tp = 0
        self.fall_fp = 0
        self.fall_fn = 0
        self.video_scores: dict[str, float] = {}

    def update_losses(
        self,
        losses: dict[str, Tensor],
        batch_size: int,
    ) -> None:
        self.samples += batch_size
        for name, value in losses.items():
            if name.startswith("loss"):
                self.loss_sums[name] += float(value.detach().cpu()) * batch_size

    def _match_detections(
        self,
        predicted_boxes: Tensor,
        predicted_scores: Tensor,
        target_boxes: Tensor,
    ) -> list[bool]:
        self.gt_boxes += int(target_boxes.shape[0])
        if predicted_boxes.shape[0] == 0:
            return []
        if target_boxes.shape[0] == 0:
            return [False] * predicted_boxes.shape[0]
        overlaps = box_iou(predicted_boxes, target_boxes.float())
        matched_targets: set[int] = set()
        true_positive_flags: list[bool] = []
        order = predicted_scores.argsort(descending=True)
        for prediction_index in order.tolist():
            values = overlaps[prediction_index].clone()
            if matched_targets:
                values[list(matched_targets)] = -1.0
            overlap, target_index = values.max(dim=0)
            is_true_positive = (
                float(overlap) >= self.match_iou_threshold
                and int(target_index) not in matched_targets
            )
            if is_true_positive:
                matched_targets.add(int(target_index))
            true_positive_flags.append(is_true_positive)
        ordered_scores = predicted_scores[order].detach().cpu().tolist()
        self.detection_records.extend(
            (float(score), flag)
            for score, flag in zip(
                ordered_scores,
                true_positive_flags,
                strict=True,
            )
        )
        return true_positive_flags

    def update_batch(
        self,
        outputs: dict[str, Tensor],
        targets: Sequence[dict[str, Tensor]],
        metadata: Sequence[dict[str, Any]],
        selected_batches: Sequence[Tensor],
    ) -> None:
        boxes = outputs["boxes"].float()
        objectness = outputs["objectness_logits"].float().sigmoid()
        stage_probabilities = outputs["stage_logits"].float().softmax(dim=-1)
        action_logits = outputs["action_logits"].float()
        onset_scores = outputs["onset_logits"].float().sigmoid()

        for batch_index, (target, sample_metadata, selected) in enumerate(
            zip(targets, metadata, selected_batches, strict=True)
        ):
            selected_boxes = boxes[batch_index, selected]
            selected_scores = objectness[batch_index, selected]
            self._match_detections(
                selected_boxes,
                selected_scores,
                target["boxes"],
            )
            report_mask = selected_scores >= self.report_person_threshold
            report_selected = selected[report_mask]
            report_boxes = selected_boxes[report_mask]

            path = str(sample_metadata["path"])
            if selected.numel() == 0:
                window_score = 0.0
            else:
                semantic_score = torch.maximum(
                    stage_probabilities[batch_index, selected, 1],
                    stage_probabilities[batch_index, selected, 2],
                )
                semantic_score = torch.maximum(
                    semantic_score,
                    onset_scores[batch_index, selected],
                )
                window_score = float(
                    (selected_scores * semantic_score).max().detach().cpu()
                )
            self.video_scores[path] = max(
                self.video_scores.get(path, 0.0),
                window_score,
            )

            actor_box_index = int(target["actor_box_index"].item())
            actor_weight = float(target["action_loss_weight"].item())
            if actor_box_index < 0 or actor_weight <= 0.0:
                continue
            self.actor_windows += 1
            target_stage = int(target["stage_label"].item())
            target_action = int(target["action_label"].item())
            target_onset = bool(float(target["onset_target"].item()) >= 0.5)

            all_predicted_fall = (
                stage_probabilities[batch_index, report_selected].argmax(dim=-1)
                .ne(0)
                .any()
                if report_selected.numel()
                else torch.tensor(False, device=boxes.device)
            )
            predicted_fall = bool(all_predicted_fall.item())
            target_fall = target_stage != 0
            if predicted_fall and target_fall:
                self.fall_tp += 1
            elif predicted_fall:
                self.fall_fp += 1
            elif target_fall:
                self.fall_fn += 1

            if report_selected.numel() == 0:
                if target_onset:
                    self.onset_fn += 1
                continue
            actor_box = target["boxes"][actor_box_index : actor_box_index + 1]
            overlaps = box_iou(report_boxes, actor_box.float()).squeeze(1)
            best_overlap, local_index = overlaps.max(dim=0)
            if float(best_overlap) < self.match_iou_threshold:
                if target_onset:
                    self.onset_fn += 1
                continue

            self.actor_detected += 1
            prediction_index = int(report_selected[int(local_index)])
            predicted_stage = int(
                stage_probabilities[batch_index, prediction_index].argmax()
            )
            predicted_action = int(
                action_logits[batch_index, prediction_index].argmax()
            )
            self.stage_confusion[target_stage][predicted_stage] += 1
            self.action_confusion[target_action][predicted_action] += 1
            self.stage_correct_detected += int(predicted_stage == target_stage)
            self.action_correct_detected += int(predicted_action == target_action)

            predicted_onset = bool(
                float(onset_scores[batch_index, prediction_index]) >= 0.5
            )
            if predicted_onset and target_onset:
                self.onset_tp += 1
            elif predicted_onset:
                self.onset_fp += 1
            elif target_onset:
                self.onset_fn += 1

    def result(self, positive_video_paths: set[str]) -> dict[str, Any]:
        losses = {
            name: total / max(self.samples, 1)
            for name, total in sorted(self.loss_sums.items())
        }
        fixed_detection = _precision_recall_at_threshold(
            self.detection_records,
            self.gt_boxes,
            self.report_person_threshold,
        )
        fall_precision = _safe_divide(
            self.fall_tp,
            self.fall_tp + self.fall_fp,
        )
        fall_recall = _safe_divide(
            self.fall_tp,
            self.fall_tp + self.fall_fn,
        )
        onset_precision = _safe_divide(
            self.onset_tp,
            self.onset_tp + self.onset_fp,
        )
        onset_recall = _safe_divide(
            self.onset_tp,
            self.onset_tp + self.onset_fn,
        )
        evaluated_paths = set(self.video_scores)
        return {
            "windows": self.samples,
            "losses": losses,
            "person_detection": {
                "ground_truth_boxes": self.gt_boxes,
                "ap50": _average_precision(
                    self.detection_records,
                    self.gt_boxes,
                ),
                "at_report_threshold": fixed_detection,
            },
            "main_actor": {
                "supervised_windows": self.actor_windows,
                "detected_iou50": self.actor_detected,
                "detection_recall50": _safe_divide(
                    self.actor_detected,
                    self.actor_windows,
                ),
                "stage_accuracy_given_detected": _safe_divide(
                    self.stage_correct_detected,
                    self.actor_detected,
                ),
                "stage_end_to_end_accuracy": _safe_divide(
                    self.stage_correct_detected,
                    self.actor_windows,
                ),
                "stage_confusion_given_detected": self.stage_confusion,
                "action_accuracy_given_detected": _safe_divide(
                    self.action_correct_detected,
                    self.actor_detected,
                ),
                "action_end_to_end_accuracy": _safe_divide(
                    self.action_correct_detected,
                    self.actor_windows,
                ),
                "action_confusion_given_detected": self.action_confusion,
            },
            "window_fall_binary": {
                "true_positives": self.fall_tp,
                "false_positives": self.fall_fp,
                "false_negatives": self.fall_fn,
                "precision": fall_precision,
                "recall": fall_recall,
                "f1": _safe_divide(
                    2.0 * fall_precision * fall_recall,
                    fall_precision + fall_recall,
                ),
            },
            "onset_at_0_5": {
                "true_positives": self.onset_tp,
                "false_positives": self.onset_fp,
                "false_negatives": self.onset_fn,
                "precision": onset_precision,
                "recall": onset_recall,
                "f1": _safe_divide(
                    2.0 * onset_precision * onset_recall,
                    onset_precision + onset_recall,
                ),
            },
            "video_fall_proxy": {
                "score_definition": (
                    "max_window(person_score * max(falling, fallen, onset))"
                ),
                **_binary_video_metrics(
                    self.video_scores,
                    positive_video_paths & evaluated_paths,
                ),
            },
        }


def _positive_video_paths(events_path: Path, video_paths: set[str]) -> set[str]:
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row["path"]).replace("\\", "/")
            for row in rows
            if str(row["path"]).replace("\\", "/") in video_paths
        }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_cli(args)
    device = _resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved = argparse.Namespace(**checkpoint["arguments"])
    _restore_saved_argument_types(saved)
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
    workers = int(saved.workers) if args.workers is None else args.workers
    loader = _make_loader(
        dataset,
        saved,
        batch_size=batch_size,
        workers=workers,
    )

    model, criterion = _make_model_and_criterion(saved)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    criterion.to(device).eval()
    amp_enabled = bool(saved.amp) and device.type == "cuda"
    amp_dtype = (
        torch.float16
        if saved.amp_dtype == "float16"
        else torch.bfloat16
    )
    if amp_dtype == torch.bfloat16 and device.type == "cuda":
        amp_enabled = amp_enabled and torch.cuda.is_bf16_supported()

    all_video_paths = {
        str(window["path"]).replace("\\", "/") for window in dataset.windows
    }
    training_mode = getattr(saved, "training_mode", "unified")
    evaluator = (
        _Evaluator(
            match_iou_threshold=args.match_iou_threshold,
            report_person_threshold=args.report_person_threshold,
        )
        if training_mode == "unified"
        else None
    )
    loss_meter = (
        _MetricAccumulator()
        if training_mode == "bbox-classifier"
        else None
    )
    classifier_meter = (
        _ClassifierMetricAccumulator()
        if training_mode == "bbox-classifier"
        else None
    )
    maximum_batches = (
        min(args.max_batches, len(loader))
        if args.max_batches > 0
        else len(loader)
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            source_images = batch["images"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            images = source_images.to(dtype=torch.float32).div_(255.0)
            history_indices = batch["history_indices"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            valid_mask = batch["valid_mask"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            actor_boxes = batch["actor_boxes"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            actor_valid_mask = batch["actor_valid_mask"].to(
                device,
                non_blocking=saved.pin_memory,
            )
            targets = _move_targets(batch["targets"], device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if training_mode == "bbox-classifier":
                    outputs = model.forward_indexed_history(
                        images,
                        actor_boxes,
                        history_indices,
                        valid_mask,
                        actor_valid_mask,
                    )
                else:
                    outputs = model.forward_indexed_history(
                        images,
                        history_indices,
                        valid_mask,
                    )
                losses = criterion(outputs, targets, batch["metadata"])
            if training_mode == "bbox-classifier":
                assert loss_meter is not None
                assert classifier_meter is not None
                loss_meter.update(losses, len(targets))
                classifier_meter.update(
                    outputs,
                    targets,
                    batch["metadata"],
                )
                completed_windows = loss_meter.sample_count
            else:
                assert evaluator is not None
                selected_batches = _select_predictions(
                    outputs,
                    person_threshold=args.person_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    pre_nms_topk=args.pre_nms_topk,
                    max_detections=args.max_detections,
                )
                evaluator.update_losses(losses, len(targets))
                evaluator.update_batch(
                    outputs,
                    targets,
                    batch["metadata"],
                    selected_batches,
                )
                completed_windows = evaluator.samples
            completed = batch_index + 1
            if (
                completed % args.log_interval == 0
                or completed == maximum_batches
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"batch={completed}/{maximum_batches} "
                    f"windows={completed_windows} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    duration = time.perf_counter() - started
    if training_mode == "bbox-classifier":
        assert loss_meter is not None
        assert classifier_meter is not None
        sample_count = loss_meter.sample_count
        evaluation_result: dict[str, Any] = {
            "mode": training_mode,
            "batches": maximum_batches,
            "duration_s": duration,
            "windows_per_s": sample_count / max(duration, 1e-9),
            **loss_meter.result(),
            **classifier_meter.result(),
        }
    else:
        assert evaluator is not None
        positive_paths = _positive_video_paths(events, all_video_paths)
        sample_count = evaluator.samples
        evaluation_result = {
            "mode": training_mode,
            "batches": maximum_batches,
            "duration_s": duration,
            "windows_per_s": sample_count / max(duration, 1e-9),
            "person_candidate_threshold": args.person_threshold,
            "person_report_threshold": args.report_person_threshold,
            "nms_iou_threshold": args.nms_iou_threshold,
            "match_iou_threshold": args.match_iou_threshold,
            **evaluator.result(positive_paths),
        }
    best_metric_name = checkpoint.get("best_metric_name", "loss")
    best_metric_value = checkpoint.get(
        "best_metric_value",
        checkpoint.get("best_val_loss", math.nan),
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "checkpoint_best_metric_name": str(best_metric_name),
        "checkpoint_best_metric_value": float(best_metric_value),
        "split": args.split,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": saved.amp_dtype,
        "dataset": {
            "root": str(dataset_root.resolve()),
            "windows": str(windows.resolve()),
            "tracks_root": str(tracks_root.resolve()),
            "actor_targets_root": str(actor_targets_root.resolve()),
            "actor_crops_root": (
                None
                if actor_crops_root is None
                else str(actor_crops_root.resolve())
            ),
            "events": str(events.resolve()),
            "total_windows": len(dataset),
            "total_videos": len(all_video_paths),
        },
        "evaluation": evaluation_result,
    }
    if best_metric_name == "loss":
        result["checkpoint_best_val_loss"] = float(best_metric_value)
    output = args.output or (
        Path("artifacts/evaluation")
        / f"{args.checkpoint.parent.name}_{args.split}_epoch"
        f"{int(checkpoint['epoch']) + 1:04d}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    result["output"] = str(output.resolve())
    return result


def main() -> None:
    result = evaluate(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
