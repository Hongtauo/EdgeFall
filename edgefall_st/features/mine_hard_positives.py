"""Mine low-scoring positive videos from training caches for HN3 replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from edgefall_st.data.labels import FALL_ACTION_ID, FALLEN_ACTION_ID
from edgefall_st.features.cache import (
    frozen_feature_collate_fn,
)
from edgefall_st.features.mine_hard_negatives import (
    _ScoredWindow,
    _chunk_positions,
    _combine,
    _dataset,
    _device,
    _load_model,
    _manifest_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path("artifacts/frozen_features/yolo26s/train"),
    )
    parser.add_argument(
        "--extra-train-cache", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/multidomain_cauca60/split_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bottom-fraction", type=float, default=0.2)
    parser.add_argument("--windows-per-video", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def _candidate_row(
    records: Sequence[_ScoredWindow],
    *,
    windows_per_video: int,
) -> dict[str, Any]:
    event_positions = [
        index
        for index, record in enumerate(records)
        if record.action_label in (FALL_ACTION_ID, FALLEN_ACTION_ID)
    ]
    positions = event_positions or list(range(len(records)))
    anchor_position = max(positions, key=lambda index: records[index].score)
    anchor = records[anchor_position]
    chunk = _chunk_positions(
        len(records), anchor_position, windows_per_video
    )
    return {
        "path": anchor.path,
        "domain": anchor.domain,
        "anchor_window_index": anchor.window_index,
        "anchor_end_time_s": anchor.end_time_s,
        "window_indices": [records[index].window_index for index in chunk],
        "video_score": max(record.score for record in records),
        "anchor_score": anchor.score,
        "action_label": anchor.action_label,
        "source": (
            "event_window_peak" if event_positions else "weak_video_peak"
        ),
    }


def _select_bottom_by_domain(
    rows: Sequence[dict[str, Any]],
    *,
    fraction: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    selected: list[dict[str, Any]] = []
    for domain in sorted(grouped):
        domain_rows = sorted(
            grouped[domain],
            key=lambda row: (float(row["video_score"]), str(row["path"])),
        )
        count = max(1, math.ceil(len(domain_rows) * fraction))
        selected.extend(domain_rows[:count])
    return sorted(
        selected,
        key=lambda row: (str(row["domain"]), float(row["video_score"])),
    )


def run_mining(args: argparse.Namespace) -> Path:
    if not 0.0 < args.bottom_fraction <= 1.0:
        raise ValueError("bottom-fraction must be in (0, 1]")
    if min(
        args.windows_per_video,
        args.batch_size,
        args.shard_cache_size,
        args.log_interval,
    ) <= 0:
        raise ValueError("Counts and sizes must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")

    selected, label_names, video_labels = _manifest_rows(args.split_manifest)
    roots = [args.train_cache, *args.extra_train_cache]
    datasets = [
        _dataset(
            root,
            selected,
            shard_cache_size=args.shard_cache_size,
            default_domain="omnifall" if index == 0 else "unknown",
        )
        for index, root in enumerate(roots)
    ]
    dataset = _combine(datasets)
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _load_model(checkpoint, dataset, device)
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
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16
    if amp_enabled and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    grouped: dict[str, list[_ScoredWindow]] = defaultdict(list)
    excluded_disagreements: set[str] = set()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            global_features = batch["global_features"].to(device, non_blocking=True)
            roi_features = batch["roi_features"].to(device, non_blocking=True)
            if device.type == "cpu":
                global_features = global_features.float()
                roi_features = roi_features.float()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(
                    global_features,
                    roi_features,
                    batch["actor_boxes"].to(device, non_blocking=True),
                    batch["valid_mask"].to(device, non_blocking=True),
                    batch["actor_valid_mask"].to(device, non_blocking=True),
                )
            scores = outputs["fall_logits"].float().sigmoid().cpu().tolist()
            for score, target, metadata in zip(
                scores, batch["targets"], batch["metadata"], strict=True
            ):
                path = str(metadata["path"])
                if not video_labels[path]:
                    continue
                if not bool(metadata["video_has_fall_event"]):
                    excluded_disagreements.add(path)
                    continue
                grouped[path].append(
                    _ScoredWindow(
                        path=path,
                        domain=str(metadata.get("domain", "unknown")),
                        window_index=int(metadata["window_index"]),
                        end_time_s=float(metadata["end_time_s"]),
                        action_label=int(target["action_label"]),
                        score=float(score),
                        label_name=label_names.get(path, ""),
                    )
                )
            if (
                (batch_index + 1) % args.log_interval == 0
                or batch_index + 1 == len(loader)
            ):
                print(
                    f"batch={batch_index + 1}/{len(loader)} "
                    f"positive_videos={len(grouped)}",
                    flush=True,
                )

    candidates = [
        _candidate_row(
            sorted(records, key=lambda value: value.end_time_s),
            windows_per_video=args.windows_per_video,
        )
        for records in grouped.values()
    ]
    selected_rows = _select_bottom_by_domain(
        candidates, fraction=args.bottom_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_counts = Counter(str(row["domain"]) for row in selected_rows)
    source_counts = Counter(str(row["source"]) for row in selected_rows)
    scores = [float(row["video_score"]) for row in selected_rows]
    summary = {
        "checkpoint": str(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "train_caches": [str(root) for root in roots],
        "selected_window_count": len(dataset),
        "eligible_positive_video_count": len(candidates),
        "excluded_label_disagreement_count": len(excluded_disagreements),
        "bottom_fraction_per_domain": args.bottom_fraction,
        "candidate_count": len(selected_rows),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "video_score": {
            "minimum": min(scores),
            "median": statistics.median(scores),
            "mean": statistics.fmean(scores),
            "maximum": max(scores),
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"output={args.output}", flush=True)
    return args.output


def main() -> None:
    run_mining(build_parser().parse_args())


if __name__ == "__main__":
    main()
