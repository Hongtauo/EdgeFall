"""Train linear probes on frozen geometry, global, and ROI cache features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from edgefall_st.evaluation.fygr import _device
from edgefall_st.evaluation.video_metrics import end_to_end_video_metrics
from edgefall_st.features.cache import (
    FrozenFeatureWindowDataset,
    MultiDomainFrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
)
from edgefall_st.training.train_fygr import (
    _cache_dataset,
    _candidate_indices,
    _combine_datasets,
    _selection_paths,
)


MODALITIES = ("geometry", "global", "roi", "visual")


def _mapping(value: str) -> tuple[str, Path]:
    domain, separator, raw_path = value.partition("=")
    if not separator or not domain.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Expected DOMAIN=CACHE")
    return domain.strip(), Path(raw_path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument(
        "--test-cache",
        type=_mapping,
        action="append",
        required=True,
        metavar="DOMAIN=CACHE",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-domain-class", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-interval", type=int, default=5000)
    return parser


FeatureDataset = FrozenFeatureWindowDataset | MultiDomainFrozenFeatureWindowDataset


def _load_parameters(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    arguments = document.get("arguments") if isinstance(document, dict) else None
    if not isinstance(arguments, dict):
        raise ValueError("Training parameters contain no arguments mapping")
    return arguments


def _path(value: Any) -> Path | None:
    return None if value in {None, ""} else Path(str(value))


def _paths(value: Any) -> list[Path]:
    return [Path(str(item)) for item in (value or [])]


def _training_datasets(
    arguments: dict[str, Any],
) -> tuple[FeatureDataset, FeatureDataset, Path]:
    manifest_path = _path(arguments.get("split_manifest"))
    if manifest_path is None:
        raise ValueError("Feature probes require a split manifest")
    selections = _selection_paths(manifest_path)
    shard_cache_size = int(arguments.get("shard_cache_size", 2))
    candidate_path = _path(arguments.get("train_candidates"))
    selected_indices = (
        None if candidate_path is None else _candidate_indices(candidate_path)
    )
    train_roots = [Path(str(arguments["train_cache"]))]
    train_roots.extend(_paths(arguments.get("extra_train_cache")))
    overlay_roots = _paths(arguments.get("train_overlay_cache"))
    train_parts = [
        _cache_dataset(
            train_roots[0],
            split="train",
            selections=selections,
            shard_cache_size=shard_cache_size,
            selected_indices=selected_indices,
        )
    ]
    train_parts.extend(
        _cache_dataset(
            root,
            split="train",
            selections=selections,
            shard_cache_size=shard_cache_size,
        )
        for root in (*train_roots[1:], *overlay_roots)
    )
    val_roots = [Path(str(arguments["val_cache"]))]
    val_roots.extend(_paths(arguments.get("extra_val_cache")))
    val_parts = [
        _cache_dataset(
            root,
            split="val",
            selections=selections,
            shard_cache_size=shard_cache_size,
        )
        for root in val_roots
    ]
    return (
        _combine_datasets(train_parts),
        _combine_datasets(val_parts),
        manifest_path,
    )


def _test_dataset(
    values: Sequence[tuple[str, Path]],
    *,
    manifest_path: Path,
    shard_cache_size: int,
) -> FeatureDataset:
    selections = _selection_paths(manifest_path)
    parts = [
        _cache_dataset(
            cache,
            split="test",
            selections=selections,
            shard_cache_size=shard_cache_size,
            default_domain=domain,
        )
        for domain, cache in values
    ]
    return _combine_datasets(parts)


def _visual_slots(dataset: FeatureDataset) -> tuple[int, ...]:
    return tuple(
        dataset.history_offsets.index(offset)
        for offset in dataset.visual_offsets
    )


def _first_person_batch(value: Tensor, *, expected_dims: int) -> Tensor:
    return value if value.ndim == expected_dims else value.select(2, 0)


def _masked_mean_and_delta_batch(values: Tensor, mask: Tensor) -> Tensor:
    values = values.float()
    mask = mask.float()
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask[:, :, None]).sum(dim=1) / denominator
    positions = torch.arange(values.shape[1], device=values.device)[None]
    first = positions.masked_fill(mask <= 0, values.shape[1]).min(dim=1).values
    last = positions.masked_fill(mask <= 0, -1).max(dim=1).values
    has_pair = mask.sum(dim=1) >= 2
    safe_first = first.clamp(max=values.shape[1] - 1)
    safe_last = last.clamp(min=0)
    rows = torch.arange(values.shape[0], device=values.device)
    delta = values[rows, safe_last] - values[rows, safe_first]
    delta = delta * has_pair[:, None]
    return torch.cat((mean, delta), dim=1)


def summarize_batch(
    batch: dict[str, Any],
    *,
    visual_slots: Sequence[int],
) -> dict[str, Tensor]:
    boxes = _first_person_batch(batch["actor_boxes"], expected_dims=3).float()
    actor_mask = _first_person_batch(
        batch["actor_valid_mask"], expected_dims=2
    ).float()
    masked_boxes = boxes * actor_mask[:, :, None]
    pair_mask = actor_mask[:, 1:] * actor_mask[:, :-1]
    differences = (masked_boxes[:, 1:] - masked_boxes[:, :-1])
    differences = differences * pair_mask[:, :, None]
    geometry = torch.cat(
        (
            masked_boxes.flatten(start_dim=1),
            differences.flatten(start_dim=1),
            actor_mask,
        ),
        dim=1,
    )

    visual_mask = batch["valid_mask"][:, list(visual_slots)].float()
    global_features = batch["global_features"].float()
    roi_features = _first_person_batch(
        batch["roi_features"], expected_dims=5
    )
    roi_features = roi_features.float().mean(dim=(-2, -1))
    global_summary = _masked_mean_and_delta_batch(
        global_features, visual_mask
    )
    roi_summary = _masked_mean_and_delta_batch(roi_features, visual_mask)
    return {
        "geometry": geometry,
        "global": global_summary,
        "roi": roi_summary,
        "visual": torch.cat((global_summary, roi_summary), dim=1),
    }


def summarize_sample(
    sample: dict[str, Any],
    *,
    visual_slots: Sequence[int],
) -> dict[str, Tensor]:
    batched = {
        name: sample[name].unsqueeze(0)
        for name in (
            "global_features",
            "roi_features",
            "actor_boxes",
            "valid_mask",
            "actor_valid_mask",
        )
    }
    return {
        name: value[0]
        for name, value in summarize_batch(
            batched, visual_slots=visual_slots
        ).items()
    }


def _loader(
    dataset: FeatureDataset | Subset,
    *,
    batch_size: int,
    workers: int,
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
        pin_memory=False,
        **options,
    )


def _sample_indices(
    dataset: FeatureDataset,
    *,
    samples_per_domain_class: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    grouped: dict[tuple[str, bool], list[int]] = {}
    for index, window in enumerate(dataset.windows):
        key = (str(window.get("domain", "omnifall")), bool(window["event_target"]))
        grouped.setdefault(key, []).append(index)
    generator = random.Random(seed)
    selected: list[int] = []
    counts: dict[str, int] = {}
    for (domain, positive), indices in sorted(grouped.items()):
        generator.shuffle(indices)
        current = indices[:samples_per_domain_class]
        selected.extend(current)
        counts[f"{domain}:{int(positive)}"] = len(current)
    selected.sort()
    return selected, counts


def _extract_training_features(
    dataset: FeatureDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    log_interval: int,
) -> tuple[dict[str, Tensor], Tensor]:
    slots = _visual_slots(dataset)
    values: dict[str, list[Tensor]] = {name: [] for name in MODALITIES}
    labels: list[Tensor] = []
    loader = _loader(
        Subset(dataset, indices), batch_size=batch_size, workers=workers
    )
    processed = 0
    for batch in loader:
        summaries = summarize_batch(batch, visual_slots=slots)
        for name in MODALITIES:
            values[name].append(summaries[name].half())
        labels.append(
            torch.stack(
                [target["event_target"] for target in batch["targets"]]
            ).float()
        )
        processed += len(batch["targets"])
        if processed % log_interval < len(batch["targets"]) or processed == len(
            indices
        ):
            print(f"extract_train={processed}/{len(indices)}", flush=True)
    return (
        {name: torch.cat(items) for name, items in values.items()},
        torch.cat(labels),
    )


def _moments(values: Tensor, *, chunk_size: int = 2048) -> tuple[Tensor, Tensor]:
    total = torch.zeros(values.shape[1], dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for start in range(0, len(values), chunk_size):
        batch = values[start : start + chunk_size].double()
        total += batch.sum(dim=0)
        squared += batch.square().sum(dim=0)
        count += len(batch)
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(1e-8)
    return mean.float(), variance.sqrt().float()


def _train_probe(
    features: Tensor,
    labels: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, Tensor, Tensor, list[float]]:
    mean, std = _moments(features)
    generator = torch.Generator().manual_seed(seed)
    model = nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    losses: list[float] = []
    for epoch in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        total_loss = 0.0
        for start in range(0, len(features), batch_size):
            indices = order[start : start + batch_size]
            batch = features[indices].to(device).float()
            batch = (batch - mean.to(device)) / std.to(device)
            target = labels[indices].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(batch).squeeze(1), target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        losses.append(total_loss / len(features))
        print(
            f"probe_epoch={epoch + 1}/{epochs} loss={losses[-1]:.6f}",
            flush=True,
        )
    model.eval()
    return model, mean, std, losses


def _evaluate_probes(
    dataset: FeatureDataset,
    probes: dict[str, nn.Linear],
    moments: dict[str, tuple[Tensor, Tensor]],
    *,
    manifest_path: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    log_interval: int,
) -> dict[str, dict[str, Any]]:
    slots = _visual_slots(dataset)
    maxima: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in MODALITIES
    }
    loader = _loader(dataset, batch_size=batch_size, workers=workers)
    processed = 0
    device_moments = {
        name: (mean.to(device), std.to(device))
        for name, (mean, std) in moments.items()
    }
    with torch.inference_mode():
        for batch in loader:
            summaries = summarize_batch(batch, visual_slots=slots)
            batch_scores: dict[str, Tensor] = {}
            for name in MODALITIES:
                mean, std = device_moments[name]
                value = summaries[name].to(device)
                value = (value - mean) / std
                batch_scores[name] = probes[name](value).sigmoid().cpu().flatten()
            for row, metadata in enumerate(batch["metadata"]):
                domain = str(metadata.get("domain", "omnifall"))
                path = str(metadata["path"])
                for name in MODALITIES:
                    score = float(batch_scores[name][row])
                    domain_scores = maxima[name].setdefault(domain, {})
                    domain_scores[path] = max(
                        domain_scores.get(path, 0.0), score
                    )
            processed += len(batch["metadata"])
            if processed % log_interval < len(batch["metadata"]) or processed == len(
                dataset
            ):
                print(
                    f"evaluate_{split}={processed}/{len(dataset)}",
                    flush=True,
                )
    result: dict[str, dict[str, Any]] = {}
    domains = sorted({domain for values in maxima.values() for domain in values})
    for name in MODALITIES:
        result[name] = {}
        for domain in domains:
            records = [
                {"path": path, "score": score}
                for path, score in maxima[name].get(domain, {}).items()
            ]
            metrics, _ = end_to_end_video_metrics(
                records,
                manifest_path=manifest_path,
                domain=domain,
                split=split,
            )
            result[name][domain] = metrics
        means = [
            float(metrics["end_to_end_video_mean_precision"])
            for metrics in result[name].values()
        ]
        result[name]["macro_end_to_end_video_mean_precision"] = sum(means) / len(
            means
        )
    return result


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if min(
        args.samples_per_domain_class,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.log_interval,
    ) <= 0:
        raise ValueError("Sampling and optimization values must be positive")
    if args.weight_decay < 0 or args.workers < 0:
        raise ValueError("weight-decay cannot be negative")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    arguments = _load_parameters(args.parameters)
    train, val, manifest_path = _training_datasets(arguments)
    test = _test_dataset(
        args.test_cache,
        manifest_path=manifest_path,
        shard_cache_size=int(arguments.get("shard_cache_size", 2)),
    )
    indices, sample_counts = _sample_indices(
        train,
        samples_per_domain_class=args.samples_per_domain_class,
        seed=args.seed,
    )
    train_features, train_labels = _extract_training_features(
        train,
        indices,
        batch_size=args.batch_size,
        workers=args.workers,
        log_interval=args.log_interval,
    )
    probes: dict[str, nn.Linear] = {}
    moments: dict[str, tuple[Tensor, Tensor]] = {}
    training_losses: dict[str, list[float]] = {}
    for modality_index, name in enumerate(MODALITIES):
        print(f"train_probe={name}", flush=True)
        model, mean, std, losses = _train_probe(
            train_features[name],
            train_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + modality_index,
            device=device,
        )
        probes[name] = model
        moments[name] = (mean, std)
        training_losses[name] = losses
    del train_features
    validation = _evaluate_probes(
        val,
        probes,
        moments,
        manifest_path=manifest_path,
        split="val",
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        log_interval=args.log_interval,
    )
    test_metrics = _evaluate_probes(
        test,
        probes,
        moments,
        manifest_path=manifest_path,
        split="test",
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        log_interval=args.log_interval,
    )
    result = {
        "parameters": str(args.parameters),
        "manifest": str(manifest_path),
        "seed": args.seed,
        "training_samples": len(indices),
        "training_sample_counts": sample_counts,
        "training_positive_fraction": float(train_labels.mean()),
        "feature_dimensions": {
            name: int(model.in_features) for name, model in probes.items()
        },
        "training_losses": training_losses,
        "validation": validation,
        "test": test_metrics,
        "definition": {
            "geometry": "flattened normalized boxes, first differences, validity",
            "global": "masked temporal mean and last-minus-first YOLO feature",
            "roi": "spatial mean then temporal mean and last-minus-first ROI feature",
            "visual": "concatenated global and ROI summaries",
            "classifier": "one standardized linear logit per modality",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    torch.save(
        {
            "arguments": vars(args),
            "models": {name: model.state_dict() for name, model in probes.items()},
            "moments": moments,
            "feature_dimensions": result["feature_dimensions"],
        },
        args.output_dir / "probes.pt",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    run_probe(build_parser().parse_args())


if __name__ == "__main__":
    main()
