"""Select FYGR checkpoints with manifest-backed validation video metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from edgefall_st.evaluation.checkpoint import resolve_checkpoint_dependency
from edgefall_st.evaluation.fygr import _check_contract, _device
from edgefall_st.evaluation.video_metrics import end_to_end_video_metrics
from edgefall_st.features.cache import (
    FrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
    roi_feature_shape,
)
from edgefall_st.models.fygr import (
    FYGR_FUSION_CONCAT_MLP,
    FrozenYoloGlobalROIConfirmer,
)


SUPPORTED_VERSIONS = {
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
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="Validation cache written as DOMAIN=PATH",
    )
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _domain_caches(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Cache must use DOMAIN=PATH syntax: {value!r}")
        domain, raw_path = value.split("=", 1)
        if not domain or not raw_path:
            raise ValueError(f"Cache must use DOMAIN=PATH syntax: {value!r}")
        if domain in seen:
            raise ValueError(f"Duplicate cache domain: {domain}")
        seen.add(domain)
        result.append((domain, Path(raw_path)))
    return result


def _load_model(
    checkpoint_path: Path,
    dataset: FrozenFeatureWindowDataset,
    device: torch.device,
) -> tuple[FrozenYoloGlobalROIConfirmer, int, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported FYGR checkpoint: {checkpoint_path}")
    _check_contract(checkpoint["feature_contract"], dataset.manifest)
    saved_args = argparse.Namespace(**checkpoint["arguments"])
    global_shape = dataset.manifest["arrays"]["global_features"]
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=resolve_checkpoint_dependency(
            saved_args.geometry_checkpoint, checkpoint_path
        ),
        history_offsets=dataset.history_offsets,
        visual_offsets=dataset.visual_offsets,
        global_feature_dim=int(global_shape[-1]),
        roi_feature_shape=roi_feature_shape(dataset.manifest),
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
    return model, int(checkpoint["epoch"]), checkpoint["feature_contract"]


def _loader(
    dataset: FrozenFeatureWindowDataset,
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    options: dict[str, Any] = {}
    if workers > 0:
        options.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        collate_fn=frozen_feature_collate_fn,
        pin_memory=pin_memory,
        **options,
    )


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    if min(
        args.batch_size,
        args.shard_cache_size,
        args.log_interval,
    ) <= 0:
        raise ValueError("Batch/cache/log values must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    domain_caches = _domain_caches(args.cache)
    checkpoint_paths = sorted(args.checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise ValueError("Checkpoint directory contains no epoch checkpoints")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    device = _device(args.device)
    first_dataset = FrozenFeatureWindowDataset(
        domain_caches[0][1], shard_cache_size=args.shard_cache_size
    )
    loaded = [
        _load_model(path, first_dataset, device) for path in checkpoint_paths
    ]
    models = [item[0] for item in loaded]
    epochs = [item[1] for item in loaded]
    feature_contracts = [item[2] for item in loaded]
    if len(set(epochs)) != len(epochs):
        raise ValueError("Checkpoint epochs are not unique")

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16
    if amp_enabled and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    summaries: dict[int, dict[str, Any]] = {
        epoch: {
            "epoch": epoch,
            "checkpoint": str(path),
            "domains": {},
        }
        for path, epoch in zip(checkpoint_paths, epochs, strict=True)
    }

    with torch.inference_mode():
        for domain, cache_path in domain_caches:
            dataset = FrozenFeatureWindowDataset(
                cache_path, shard_cache_size=args.shard_cache_size
            )
            for feature_contract in feature_contracts:
                _check_contract(feature_contract, dataset.manifest)
            split = str(dataset.manifest.get("split", ""))
            if not split:
                raise ValueError(f"Cache has no split metadata: {cache_path}")
            loader = _loader(
                dataset,
                batch_size=args.batch_size,
                workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            maxima: list[dict[str, float]] = [dict() for _ in models]
            for batch_index, batch in enumerate(loader):
                global_features = batch["global_features"].to(
                    device, non_blocking=True
                )
                roi_features = batch["roi_features"].to(
                    device, non_blocking=True
                )
                if device.type == "cpu":
                    global_features = global_features.float()
                    roi_features = roi_features.float()
                actor_boxes = batch["actor_boxes"].to(device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True)
                actor_valid_mask = batch["actor_valid_mask"].to(
                    device, non_blocking=True
                )
                paths = [str(metadata["path"]) for metadata in batch["metadata"]]
                for model_index, model in enumerate(models):
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
                    scores = outputs["fall_logits"].float().sigmoid().cpu()
                    current = maxima[model_index]
                    for path, score in zip(paths, scores, strict=True):
                        current[path] = max(current.get(path, 0.0), float(score))
                if (
                    (batch_index + 1) % args.log_interval == 0
                    or batch_index + 1 == len(loader)
                ):
                    print(
                        f"domain={domain} batch={batch_index + 1}/{len(loader)}",
                        flush=True,
                    )
            for model_index, epoch in enumerate(epochs):
                records = [
                    {"path": path, "score": score}
                    for path, score in maxima[model_index].items()
                ]
                metrics, _ = end_to_end_video_metrics(
                    records,
                    manifest_path=args.video_manifest,
                    domain=domain,
                    split=split,
                )
                summaries[epoch]["domains"][domain] = metrics

    metric_name = "end_to_end_video_mean_precision"
    for summary in summaries.values():
        values = [
            float(metrics[metric_name])
            for metrics in summary["domains"].values()
        ]
        summary["macro_end_to_end_video_mean_precision"] = sum(values) / len(
            values
        )
    ranked = sorted(
        summaries.values(),
        key=lambda item: (
            -float(item["macro_end_to_end_video_mean_precision"]),
            int(item["epoch"]),
        ),
    )
    result = {
        "selection_metric": "macro_end_to_end_video_mean_precision",
        "video_manifest": str(args.video_manifest),
        "domains": [domain for domain, _ in domain_caches],
        "best": ranked[0],
        "epochs": [summaries[epoch] for epoch in sorted(summaries)],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "selection.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        "best_epoch={epoch} best_metric={metric:.6f} output={output}".format(
            epoch=ranked[0]["epoch"],
            metric=ranked[0]["macro_end_to_end_video_mean_precision"],
            output=output_path,
        ),
        flush=True,
    )
    return result


def main() -> None:
    run_selection(build_parser().parse_args())


if __name__ == "__main__":
    main()
