"""Evaluate a frozen-YOLO global/ROI confirmer feature cache."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from edgefall_st.evaluation.checkpoint import resolve_checkpoint_dependency
from edgefall_st.evaluation.video_metrics import (
    end_to_end_video_metrics as _end_to_end_video_metrics,
)
from edgefall_st.features.cache import (
    FrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
    roi_feature_shape,
)
from edgefall_st.models.fygr import (
    FYGR_FUSION_CONCAT_MLP,
    FrozenYoloGlobalROIConfirmer,
)
from edgefall_st.training.train import (
    _ClassifierMetricAccumulator,
    _MetricAccumulator,
    _move_targets,
)
from edgefall_st.training.train_fygr import (
    _FusionStabilityAccumulator,
    _criterion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--domain")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _check_contract(saved: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "format",
        "history_offsets",
        "visual_offsets",
        "visual_input_size",
        "geometry_input_size",
        "weights_sha256",
        "feature_adapter",
    )
    mismatch = [key for key in keys if saved.get(key) != current.get(key)]
    if saved.get("arrays", {}).get("global_features") != current.get(
        "arrays", {}
    ).get("global_features"):
        mismatch.append("arrays.global_features")
    if roi_feature_shape(saved) != roi_feature_shape(current):
        mismatch.append("arrays.roi_features")
    if mismatch:
        raise ValueError(
            "Evaluation cache does not match checkpoint feature contract: "
            + ", ".join(mismatch)
        )


def run_evaluation(args: argparse.Namespace) -> dict[str, float]:
    if min(args.batch_size, args.shard_cache_size, args.log_interval) <= 0:
        raise ValueError("Batch/cache/log values must be positive")
    if args.workers < 0 or args.max_batches < 0:
        raise ValueError("workers/max-batches cannot be negative")
    if args.video_manifest is not None and args.max_batches:
        raise ValueError("End-to-end video metrics require a complete evaluation")
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_version = checkpoint.get("version")
    if checkpoint_version not in {
        "v3.4-fygr",
        "v3.5-md-fygr",
        "v3.6-gcva",
        "v3.6-md-gcva",
        "v3.7-person-mil",
        "v3.7-md-person-mil",
        "v3.8-temporal-token",
        "v3.8-md-temporal-token",
        "v3.8-person-mil-temporal-token",
        "v3.8-md-person-mil-temporal-token",
        "v3.9-summary-ranking",
        "v3.9-md-summary-ranking",
        "v3.10-hard-tail-ranking",
        "v3.10-md-hard-tail-ranking",
    }:
        raise ValueError("Checkpoint is not a supported FYGR checkpoint")
    saved_args = argparse.Namespace(**checkpoint["arguments"])
    dataset = FrozenFeatureWindowDataset(
        args.cache, shard_cache_size=args.shard_cache_size
    )
    _check_contract(checkpoint["feature_contract"], dataset.manifest)
    global_shape = dataset.manifest["arrays"]["global_features"]
    roi_shape = roi_feature_shape(dataset.manifest)
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=resolve_checkpoint_dependency(
            saved_args.geometry_checkpoint, args.checkpoint
        ),
        history_offsets=dataset.history_offsets,
        visual_offsets=dataset.visual_offsets,
        global_feature_dim=int(global_shape[-1]),
        roi_feature_shape=roi_shape,
        temporal_dim=int(saved_args.temporal_dim),
        dropout=float(saved_args.dropout),
        use_global=bool(saved_args.use_global),
        use_roi=bool(saved_args.use_roi),
        fusion_mode=str(
            getattr(saved_args, "fusion_mode", FYGR_FUSION_CONCAT_MLP)
        ),
        attention_heads=int(getattr(saved_args, "attention_heads", 4)),
        attention_ffn_dim=int(
            getattr(saved_args, "attention_ffn_dim", 256)
        ),
        person_mil_temperature=float(
            getattr(saved_args, "person_mil_temperature", 0.0)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    criterion = _criterion(saved_args).to(device)
    options: dict[str, Any] = {}
    if args.workers > 0:
        options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        collate_fn=frozen_feature_collate_fn,
        pin_memory=device.type == "cuda",
        **options,
    )
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16
    if amp_enabled and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    meter = _MetricAccumulator()
    classifier_meter = _ClassifierMetricAccumulator()
    fusion_meter = _FusionStabilityAccumulator(
        use_global=model.use_global,
        use_roi=model.use_roi,
        visual_slots=len(model.visual_offsets),
    )
    records: list[dict[str, Any]] = []
    total_batches = (
        min(len(loader), args.max_batches) if args.max_batches else len(loader)
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= total_batches:
                break
            global_features = batch["global_features"].to(device, non_blocking=True)
            roi_features = batch["roi_features"].to(device, non_blocking=True)
            if device.type == "cpu":
                global_features = global_features.float()
                roi_features = roi_features.float()
            actor_boxes = batch["actor_boxes"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            actor_valid_mask = batch["actor_valid_mask"].to(
                device, non_blocking=True
            )
            targets = _move_targets(batch["targets"], device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(
                    global_features,
                    roi_features,
                    actor_boxes,
                    valid_mask,
                    actor_valid_mask,
                )
                losses = criterion(outputs, targets, batch["metadata"])
            meter.update(losses, len(targets))
            classifier_meter.update(outputs, targets, batch["metadata"])
            fusion_meter.update(outputs)
            fall_scores = outputs["fall_logits"].float().sigmoid().cpu()
            geometry_scores = (
                outputs["geometry_fall_logits"].float().sigmoid().cpu()
            )
            global_scores = (
                outputs["global_fall_logits"].float().sigmoid().cpu()
                if "global_fall_logits" in outputs
                else None
            )
            roi_scores = (
                outputs["roi_fall_logits"].float().sigmoid().cpu()
                if "roi_fall_logits" in outputs
                else None
            )
            attention_weights = (
                outputs["fusion_attention_weights"].float().cpu()
                if "fusion_attention_weights" in outputs
                else None
            )
            person_scores = outputs.get("person_fall_logits")
            if person_scores is not None:
                person_scores = person_scores.float().sigmoid().cpu()
            person_valid = outputs.get("person_valid_mask")
            if person_valid is not None:
                person_valid = person_valid.cpu()
            attention_index = 0
            global_attention = None
            roi_attention = None
            current_attention = attention_weights
            attention_slots = 1
            if current_attention is not None and current_attention.ndim == 3:
                current_attention = current_attention[:, -1]
                attention_slots = len(model.visual_offsets)
            if attention_weights is not None and model.use_global:
                global_attention = current_attention[
                    :, attention_index : attention_index + attention_slots
                ].sum(dim=1)
                attention_index += attention_slots
            if attention_weights is not None and model.use_roi:
                roi_attention = current_attention[
                    :, attention_index : attention_index + attention_slots
                ].sum(dim=1)
            for row, (target, metadata) in enumerate(
                zip(targets, batch["metadata"], strict=True)
            ):
                records.append(
                    {
                        "path": metadata["path"],
                        "window_index": metadata["window_index"],
                        "end_time_s": metadata["end_time_s"],
                        "event_target": int(bool(metadata["event_target"])),
                        "stage_label": int(target["stage_label"].cpu()),
                        "action_label": int(target["action_label"].cpu()),
                        "score": float(fall_scores[row]),
                        "geometry_score": float(geometry_scores[row]),
                        "global_score": (
                            ""
                            if global_scores is None
                            else float(global_scores[row])
                        ),
                        "roi_score": (
                            "" if roi_scores is None else float(roi_scores[row])
                        ),
                        "global_attention": (
                            ""
                            if global_attention is None
                            else float(global_attention[row])
                        ),
                        "roi_attention": (
                            ""
                            if roi_attention is None
                            else float(roi_attention[row])
                        ),
                        "selected_person_index": int(
                            outputs["selected_person_index"][row].cpu()
                        ),
                        "person_scores": json.dumps(
                            [
                                float(score)
                                for person_index, score in enumerate(
                                    person_scores[row]
                                )
                                if person_valid is None
                                or bool(person_valid[row, person_index])
                            ]
                            if person_scores is not None
                            else [],
                            ensure_ascii=False,
                        ),
                    }
                )
            if (
                (batch_index + 1) % args.log_interval == 0
                or batch_index + 1 == total_batches
            ):
                print(f"batch={batch_index + 1}/{total_batches}", flush=True)
    metrics = meter.result()
    metrics.update(classifier_meter.result())
    metrics.update(fusion_meter.result())
    missing_videos: list[dict[str, Any]] = []
    if args.video_manifest is not None:
        domain = args.domain or dataset.manifest.get("domain")
        split = dataset.manifest.get("split")
        if not domain or not split:
            raise ValueError(
                "--domain and a cache split are required for video metrics"
            )
        end_to_end, missing_videos = _end_to_end_video_metrics(
            records,
            manifest_path=args.video_manifest,
            domain=str(domain),
            split=str(split),
        )
        metrics.update(end_to_end)
    version_names = {
        "v3.4-fygr": "v34",
        "v3.5-md-fygr": "v35_md",
        "v3.6-gcva": "v36_gcva",
        "v3.6-md-gcva": "v36_md_gcva",
        "v3.7-person-mil": "v37_person_mil",
        "v3.7-md-person-mil": "v37_md_person_mil",
        "v3.8-temporal-token": "v38_temporal_token",
        "v3.8-md-temporal-token": "v38_md_temporal_token",
        "v3.8-person-mil-temporal-token": "v38_person_mil_temporal_token",
        "v3.8-md-person-mil-temporal-token": (
            "v38_md_person_mil_temporal_token"
        ),
        "v3.9-summary-ranking": "v39_summary_ranking",
        "v3.9-md-summary-ranking": "v39_md_summary_ranking",
        "v3.10-hard-tail-ranking": "v310_hard_tail_ranking",
        "v3.10-md-hard-tail-ranking": "v310_md_hard_tail_ranking",
    }
    version_name = version_names[str(checkpoint_version)]
    output_dir = args.output_dir or Path("artifacts/evaluations") / (
        f"{version_name}_{dataset.manifest.get('split', 'data')}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.video_manifest is not None:
        with (output_dir / "missing_videos.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("path", "label", "assigned_score"),
            )
            writer.writeheader()
            writer.writerows(missing_videos)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"output_dir={output_dir}", flush=True)
    return metrics


def main() -> None:
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()
