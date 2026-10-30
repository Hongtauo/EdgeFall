"""Mine semantic and V3.6 score-based hard-negative training chunks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from edgefall_st.evaluation.fygr import _check_contract
from edgefall_st.features.cache import (
    FrozenFeatureWindowDataset,
    MultiDomainFrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
)
from edgefall_st.models.fygr import (
    FYGR_FUSION_CONCAT_MLP,
    FrozenYoloGlobalROIConfirmer,
)
from edgefall_st.training.hard_negatives import (
    HARD_NEGATIVE_SCORE,
    HARD_NEGATIVE_SEMANTIC,
)


OMNIFALL_HARD_CATEGORIES = {
    3: "sit",
    4: "sit",
    5: "lie_down",
    6: "lie_down",
    10: "kneel_squat",
    11: "kneel_squat",
    12: "kneel_squat",
    13: "kneel_squat",
}
CAUCA_HARD_CATEGORIES = {
    "Sit down": "sit",
    "Kneel": "kneel_squat",
    "Pick up object": "bend_pick",
}


@dataclass(frozen=True)
class _ScoredWindow:
    path: str
    domain: str
    window_index: int
    end_time_s: float
    action_label: int
    score: float
    label_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/checkpoints/20260730_231605_455025/best.pt"
        ),
    )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/hard_negatives/v36_cauca60_yolo26s/train.jsonl"
        ),
    )
    parser.add_argument("--windows-per-video", type=int, default=12)
    parser.add_argument("--semantic-chunks-per-video", type=int, default=2)
    parser.add_argument("--score-chunks-per-video", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _manifest_rows(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, bool]]:
    selected: dict[str, set[str]] = defaultdict(set)
    label_names: dict[str, str] = {}
    video_labels: dict[str, bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["split"]) != "train":
                continue
            domain = str(row["domain"])
            video_path = str(row["path"])
            selected[domain].add(video_path)
            label_names[video_path] = str(row["label_name"])
            video_labels[video_path] = bool(int(row["label"]))
    if not selected:
        raise ValueError("Split manifest contains no training rows")
    return dict(selected), label_names, video_labels


def _dataset(
    root: Path,
    selected: dict[str, set[str]],
    *,
    shard_cache_size: int,
    default_domain: str,
) -> FrozenFeatureWindowDataset:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("split", "")) != "train":
        raise ValueError(f"Hard-negative mining requires train cache: {root}")
    domain = str(manifest.get("domain", default_domain))
    paths = selected.get(domain)
    if paths is None:
        raise ValueError(f"No training selection for {domain}/{root}")
    return FrozenFeatureWindowDataset(
        root,
        shard_cache_size=shard_cache_size,
        selected_paths=paths,
        domain=domain,
    )


def _combine(
    datasets: Sequence[FrozenFeatureWindowDataset],
) -> FrozenFeatureWindowDataset | MultiDomainFrozenFeatureWindowDataset:
    if len(datasets) == 1:
        return datasets[0]
    return MultiDomainFrozenFeatureWindowDataset(datasets)


def _load_model(
    checkpoint: dict[str, Any],
    dataset: FrozenFeatureWindowDataset | MultiDomainFrozenFeatureWindowDataset,
    device: torch.device,
) -> FrozenYoloGlobalROIConfirmer:
    _check_contract(checkpoint["feature_contract"], dataset.manifest)
    saved = argparse.Namespace(**checkpoint["arguments"])
    arrays = dataset.manifest["arrays"]
    model = FrozenYoloGlobalROIConfirmer(
        geometry_checkpoint=Path(saved.geometry_checkpoint),
        history_offsets=dataset.history_offsets,
        visual_offsets=dataset.visual_offsets,
        global_feature_dim=int(arrays["global_features"][-1]),
        roi_feature_shape=tuple(int(value) for value in arrays["roi_features"]),
        temporal_dim=int(saved.temporal_dim),
        dropout=float(saved.dropout),
        use_global=bool(saved.use_global),
        use_roi=bool(saved.use_roi),
        fusion_mode=str(
            getattr(saved, "fusion_mode", FYGR_FUSION_CONCAT_MLP)
        ),
        attention_heads=int(getattr(saved, "attention_heads", 4)),
        attention_ffn_dim=int(
            getattr(saved, "attention_ffn_dim", 256)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _hard_category(window: _ScoredWindow) -> str | None:
    if window.domain == "omnifall":
        return OMNIFALL_HARD_CATEGORIES.get(window.action_label)
    if window.domain == "caucafall":
        return CAUCA_HARD_CATEGORIES.get(window.label_name)
    return None


def _chunk_positions(length: int, anchor: int, width: int) -> range:
    if length <= width:
        return range(length)
    start = min(max(anchor - width // 2, 0), length - width)
    return range(start, start + width)


def _select_non_overlapping(
    records: Sequence[_ScoredWindow],
    candidates: Sequence[int],
    *,
    count: int,
    width: int,
) -> list[int]:
    selected: list[int] = []
    occupied: set[int] = set()
    for position in sorted(
        candidates,
        key=lambda value: records[value].score,
        reverse=True,
    ):
        chunk = set(_chunk_positions(len(records), position, width))
        if chunk.intersection(occupied):
            continue
        selected.append(position)
        occupied.update(chunk)
        if len(selected) == count:
            break
    return selected


def _semantic_positions(
    records: Sequence[_ScoredWindow],
    *,
    count: int,
    width: int,
) -> list[tuple[int, str]]:
    clip_category = _hard_category(records[0])
    if records[0].domain == "caucafall" and clip_category is not None:
        return [
            (position, clip_category)
            for position in _select_non_overlapping(
                records,
                list(range(len(records))),
                count=count,
                width=width,
            )
        ]
    candidates: list[tuple[int, str]] = []
    run_start = 0
    while run_start < len(records):
        category = _hard_category(records[run_start])
        run_end = run_start + 1
        while (
            run_end < len(records)
            and _hard_category(records[run_end]) == category
        ):
            run_end += 1
        if category is not None:
            peak = max(
                range(run_start, run_end),
                key=lambda position: records[position].score,
            )
            candidates.append((peak, category))
        run_start = run_end
    if not candidates:
        return []
    category_by_position = dict(candidates)
    selected = _select_non_overlapping(
        records,
        [position for position, _ in candidates],
        count=count,
        width=width,
    )
    return [(position, category_by_position[position]) for position in selected]


def _candidate_row(
    records: Sequence[_ScoredWindow],
    position: int,
    *,
    kind: str,
    category: str,
    width: int,
    source: str,
    confidence: str,
) -> dict[str, Any]:
    anchor = records[position]
    chunk = _chunk_positions(len(records), position, width)
    return {
        "path": anchor.path,
        "domain": anchor.domain,
        "kind": kind,
        "category": category,
        "anchor_window_index": anchor.window_index,
        "anchor_end_time_s": anchor.end_time_s,
        "window_indices": [records[index].window_index for index in chunk],
        "baseline_score": anchor.score,
        "action_label": anchor.action_label,
        "label_name": anchor.label_name,
        "source": source,
        "confidence": confidence,
    }


def _build_candidates(
    records: Sequence[_ScoredWindow],
    *,
    windows_per_video: int,
    semantic_count: int,
    score_count: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    semantic = _semantic_positions(
        records,
        count=semantic_count,
        width=windows_per_video,
    )
    rows: list[dict[str, Any]] = []
    for position, category in semantic:
        domain = records[position].domain
        rows.append(
            _candidate_row(
                records,
                position,
                kind=HARD_NEGATIVE_SEMANTIC,
                category=category,
                width=windows_per_video,
                source=(
                    "frame_action_baseline_peak"
                    if domain == "omnifall"
                    else "clip_action_baseline_peak"
                ),
                confidence="strong" if domain == "omnifall" else "clip",
            )
        )
    if semantic:
        return rows
    score_positions = _select_non_overlapping(
        records,
        [
            index
            for index, record in enumerate(records)
            if record.score >= score_threshold
        ],
        count=score_count,
        width=windows_per_video,
    )
    rows.extend(
        _candidate_row(
            records,
            position,
            kind=HARD_NEGATIVE_SCORE,
            category="score_mined_unknown",
            width=windows_per_video,
            source="baseline_peak",
            confidence="binary_negative",
        )
        for position in score_positions
    )
    return rows


def run_mining(args: argparse.Namespace) -> Path:
    if min(
        args.windows_per_video,
        args.semantic_chunks_per_video,
        args.score_chunks_per_video,
        args.batch_size,
        args.shard_cache_size,
        args.log_interval,
    ) <= 0:
        raise ValueError("Counts and sizes must be positive")
    if args.workers < 0 or not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("Invalid worker count or score threshold")
    selected, label_names, video_labels = _manifest_rows(
        args.split_manifest
    )
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
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
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
    cache_label_disagreements: set[str] = set()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            global_features = batch["global_features"].to(
                device, non_blocking=True
            )
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
                    batch["actor_valid_mask"].to(
                        device, non_blocking=True
                    ),
                )
            scores = outputs["fall_logits"].float().sigmoid().cpu().tolist()
            for score, target, metadata in zip(
                scores,
                batch["targets"],
                batch["metadata"],
                strict=True,
            ):
                path = str(metadata["path"])
                manifest_label = video_labels[path]
                if bool(metadata["video_has_fall_event"]) != manifest_label:
                    cache_label_disagreements.add(path)
                # Hard negatives follow the locked video split label. A
                # positive video without a visible transition is not a
                # negative mining candidate.
                if manifest_label:
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
                    f"negative_videos={len(grouped)}",
                    flush=True,
                )

    candidates: list[dict[str, Any]] = []
    negative_scores: list[float] = []
    for path in sorted(grouped, key=str.casefold):
        records = sorted(grouped[path], key=lambda value: value.end_time_s)
        negative_scores.extend(record.score for record in records)
        candidates.extend(
            _build_candidates(
                records,
                windows_per_video=args.windows_per_video,
                semantic_count=args.semantic_chunks_per_video,
                score_count=args.score_chunks_per_video,
                score_threshold=args.score_threshold,
            )
        )
    candidates.sort(
        key=lambda row: (
            str(row["kind"]),
            str(row["category"]),
            str(row["path"]).casefold(),
            int(row["anchor_window_index"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    kind_counts = Counter(str(row["kind"]) for row in candidates)
    category_counts = Counter(str(row["category"]) for row in candidates)
    domain_counts = Counter(str(row["domain"]) for row in candidates)
    candidate_videos = {str(row["path"]) for row in candidates}
    summary = {
        "checkpoint": str(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "train_caches": [str(root) for root in roots],
        "selected_window_count": len(dataset),
        "negative_video_count": len(grouped),
        "cache_video_label_disagreement_count": len(
            cache_label_disagreements
        ),
        "candidate_count": len(candidates),
        "candidate_video_count": len(candidate_videos),
        "negative_video_without_candidate_count": len(grouped)
        - len(candidate_videos),
        "kind_counts": dict(sorted(kind_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "windows_per_video": args.windows_per_video,
        "semantic_chunks_per_video": args.semantic_chunks_per_video,
        "score_chunks_per_video": args.score_chunks_per_video,
        "score_threshold": args.score_threshold,
        "negative_window_score": {
            "minimum": min(negative_scores),
            "median": statistics.median(negative_scores),
            "mean": statistics.fmean(negative_scores),
            "maximum": max(negative_scores),
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"hard_negative_manifest={args.output}", flush=True)
    return args.output


def main() -> None:
    run_mining(build_parser().parse_args())


if __name__ == "__main__":
    main()
