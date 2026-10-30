"""Mine V3.3-G high-score and semantic hard negatives for V3.4."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from edgefall_st.features.cache import (
    FrozenFeatureWindowDataset,
    frozen_feature_collate_fn,
)
from edgefall_st.models.fygr import load_frozen_v33_geometry
from edgefall_st.training.bbox_criterion import CONTROLLED_DESCENT_ACTION_IDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/frozen_features/yolo26x/train"),
    )
    parser.add_argument(
        "--geometry-checkpoint",
        type=Path,
        default=Path(
            "artifacts/checkpoints/20260725_220646_947213/best.pt"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--geometry-threshold", type=float, default=0.25)
    parser.add_argument(
        "--hard-action-ids",
        nargs="+",
        type=int,
        default=CONTROLLED_DESCENT_ACTION_IDS,
    )
    parser.add_argument("--easy-negative-keep", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def run_mining(args: argparse.Namespace) -> Path:
    if not 0.0 <= args.geometry_threshold <= 1.0:
        raise ValueError("geometry-threshold must be in [0, 1]")
    if not 0.0 <= args.easy_negative_keep <= 1.0:
        raise ValueError("easy-negative-keep must be in [0, 1]")
    if min(args.batch_size, args.log_interval) <= 0 or args.workers < 0:
        raise ValueError("Invalid batch/worker/log configuration")
    dataset = FrozenFeatureWindowDataset(args.cache)
    if dataset.manifest.get("split") != "train":
        raise ValueError(
            "Candidate mining is restricted to a cache marked split=train"
        )
    device = _device(args.device)
    model = load_frozen_v33_geometry(
        args.geometry_checkpoint, history_offsets=dataset.history_offsets
    ).to(device)
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
    output = args.output or args.cache / "candidates_v33g.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(args.seed)
    hard_actions = {int(value) for value in args.hard_action_ids}
    counts: dict[str, int] = {}
    cache_index = 0
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            actor_boxes = batch["actor_boxes"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            actor_valid_mask = batch["actor_valid_mask"].to(
                device, non_blocking=True
            )
            outputs = model.forward_geometry_history(
                actor_boxes, valid_mask, actor_valid_mask
            )
            scores = outputs["fall_logits"].float().sigmoid().cpu().tolist()
            for score, target, metadata in zip(
                scores,
                batch["targets"],
                batch["metadata"],
                strict=True,
            ):
                event_target = bool(metadata["event_target"])
                action_label = int(target["action_label"])
                if event_target:
                    category = "positive"
                    selected = True
                elif float(score) >= args.geometry_threshold:
                    category = "geometry_high_negative"
                    selected = True
                elif action_label in hard_actions:
                    category = "semantic_hard_negative"
                    selected = True
                else:
                    category = "easy_negative"
                    selected = random_generator.random() < args.easy_negative_keep
                counts[category] = counts.get(category, 0) + 1
                if selected:
                    counts["selected"] = counts.get("selected", 0) + 1
                handle.write(
                    json.dumps(
                        {
                            "cache_index": cache_index,
                            "window_index": int(metadata["window_index"]),
                            "path": str(metadata["path"]),
                            "event_target": event_target,
                            "action_label": action_label,
                            "geometry_score": float(score),
                            "category": category,
                            "selected": selected,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                cache_index += 1
            if (batch_index + 1) % args.log_interval == 0:
                print(
                    f"batch={batch_index + 1}/{len(loader)} windows={cache_index}",
                    flush=True,
                )
    summary = {
        "cache": str(args.cache),
        "geometry_checkpoint": str(args.geometry_checkpoint),
        "geometry_threshold": args.geometry_threshold,
        "hard_action_ids": sorted(hard_actions),
        "easy_negative_keep": args.easy_negative_keep,
        "seed": args.seed,
        "counts": counts,
    }
    with output.with_suffix(".summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"candidates={output}", flush=True)
    return output


def main() -> None:
    run_mining(build_parser().parse_args())


if __name__ == "__main__":
    main()
