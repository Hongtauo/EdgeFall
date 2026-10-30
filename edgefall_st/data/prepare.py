"""Prepare deterministic OmniFall annotations and manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from edgefall_st.data.annotations import (
    VideoAnnotations,
    clean_annotations,
    load_annotations,
    write_clean_annotations,
)
from edgefall_st.data.labels import ACTION_LABELS
from edgefall_st.data.split import (
    SplitRatios,
    stratified_video_split,
    write_split_manifest,
)
from edgefall_st.data.windows import FallEvent, WindowLabel, build_fall_events, build_windows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_events(path: Path, events: Iterable[FallEvent]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "start_s",
        "end_s",
        "falling_end_s",
        "includes_fallen",
    ]
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event.as_dict())
            count += 1
    return count


def _validate_label_map(label_map_path: Path) -> None:
    with label_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        observed = {int(row["id"]): row["label"] for row in rows}
    if observed != ACTION_LABELS:
        raise ValueError(
            f"{label_map_path} does not match the canonical 16-class taxonomy: "
            f"{observed}"
        )


def _validate_video_inventory(
    dataset_root: Path,
    videos: Iterable[VideoAnnotations],
) -> dict[str, int]:
    referenced = {video.path for video in videos}
    available = {
        path.relative_to(dataset_root).with_suffix("").as_posix()
        for path in dataset_root.rglob("*.mp4")
    }
    missing = sorted(referenced - available)
    unreferenced = sorted(available - referenced)
    if missing or unreferenced:
        raise ValueError(
            "Annotation/video inventory mismatch: "
            f"missing={len(missing)}, unreferenced={len(unreferenced)}; "
            f"examples={missing[:3] + unreferenced[:3]}"
        )
    return {
        "referenced_videos": len(referenced),
        "available_videos": len(available),
        "missing_videos": 0,
        "unreferenced_videos": 0,
    }


def _counted_window_rows(
    videos: Iterable[VideoAnnotations],
    *,
    effective_fps: float,
    window_length: int,
    stride: int,
    action_counts: Counter[int],
    stage_counts: Counter[int],
) -> Iterator[dict[str, object]]:
    for video in videos:
        for window in build_windows(
            video,
            effective_fps=effective_fps,
            window_length=window_length,
            stride=stride,
        ):
            action_counts[window.action_label] += 1
            stage_counts[window.stage_label] += 1
            yield window.as_dict()


def prepare_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    effective_fps: float = 10.0,
    window_length: int = 32,
    stride: int = 4,
    boundary_fps: float = 30.0,
    seed: int = 20260717,
) -> dict[str, object]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    source_csv = dataset_root / "of-syn.csv"
    _validate_label_map(dataset_root / "label2id.csv")

    segments, metadata_fields = load_annotations(source_csv)
    videos, cleaning_report = clean_annotations(
        segments,
        boundary_fps=boundary_fps,
    )
    inventory = _validate_video_inventory(dataset_root, videos)
    assignments = stratified_video_split(
        videos,
        ratios=SplitRatios(),
        seed=seed,
    )

    write_clean_annotations(
        videos,
        output_dir / "annotations_clean.csv",
        metadata_fields,
    )
    _write_json(output_dir / "cleaning_report.json", cleaning_report.as_dict())
    write_split_manifest(
        videos,
        assignments,
        output_dir / "split_manifest.csv",
    )

    event_count = _write_events(
        output_dir / "events.csv",
        (
            event
            for video in videos
            for event in build_fall_events(video)
        ),
    )

    split_window_counts: dict[str, int] = {}
    split_action_counts: dict[str, dict[str, int]] = {}
    split_stage_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_videos = [
            video for video in videos if assignments[video.path] == split
        ]
        action_counts: Counter[int] = Counter()
        stage_counts: Counter[int] = Counter()
        rows = _counted_window_rows(
            split_videos,
            effective_fps=effective_fps,
            window_length=window_length,
            stride=stride,
            action_counts=action_counts,
            stage_counts=stage_counts,
        )
        split_window_counts[split] = _write_jsonl(
            output_dir / f"windows_{split}.jsonl",
            rows,
        )
        split_action_counts[split] = {
            str(label): count for label, count in sorted(action_counts.items())
        }
        split_stage_counts[split] = {
            str(label): count for label, count in sorted(stage_counts.items())
        }

    split_counts = Counter(assignments.values())
    summary: dict[str, object] = {
        "dataset_root": str(dataset_root.resolve()),
        "source_csv": str(source_csv.resolve()),
        "inventory": inventory,
        "videos": len(videos),
        "segments": cleaning_report.output_rows,
        "events": event_count,
        "split_videos": dict(sorted(split_counts.items())),
        "split_windows": split_window_counts,
        "split_action_windows": split_action_counts,
        "split_stage_windows": split_stage_counts,
        "effective_fps": effective_fps,
        "window_length": window_length,
        "stride": stride,
        "boundary_fps": boundary_fps,
        "seed": seed,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/omnifall_v1"),
    )
    parser.add_argument("--effective-fps", type=float, default=10.0)
    parser.add_argument("--window-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--boundary-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(
        args.dataset_root,
        args.output_dir,
        effective_fps=args.effective_fps,
        window_length=args.window_length,
        stride=args.stride,
        boundary_fps=args.boundary_fps,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
