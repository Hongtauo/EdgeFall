"""Generate label-blind YOLO actor boxes for selected multi-domain videos."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from edgefall_st.pseudo.actor_targets import (
    ActorSelectionQuality,
    load_track_frames,
    select_primary_actor,
)
from edgefall_st.pseudo.batched_tracker import BatchVideoTask, track_video_batch
from edgefall_st.pseudo.yolo26_tracker import TeacherConfig


@dataclass(frozen=True)
class ManifestVideo:
    path: str
    video_path: Path
    split: str
    label: bool


def _artifact_path(root: Path, sample_id: str, suffix: str) -> Path:
    relative = Path(*sample_id.split("/"))
    return root / relative.parent / f"{relative.name}{suffix}"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _records(args: argparse.Namespace) -> list[ManifestVideo]:
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        ManifestVideo(
            path=str(row["path"]),
            video_path=Path(row["video_path"]),
            split=str(row["split"]),
            label=bool(int(row["label"])),
        )
        for row in rows
        if row["domain"] == args.domain and row["split"] in args.splits
    ]
    if not selected:
        raise ValueError("The manifest selected no videos")
    missing = [str(row.video_path) for row in selected if not row.video_path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sorted(selected, key=lambda row: row.path.casefold())


def _build_actor_boxes(
    record: ManifestVideo,
    output_root: Path,
    *,
    overwrite: bool,
) -> ActorSelectionQuality | None:
    track_path = _artifact_path(output_root / "tracks", record.path, ".jsonl")
    target_path = _artifact_path(
        output_root / "actor_boxes", record.path, ".jsonl"
    )
    quality_path = _artifact_path(
        output_root / "actor_quality", record.path, ".json"
    )
    if target_path.exists() and quality_path.exists() and not overwrite:
        return None
    frames = load_track_frames(track_path)
    targets, quality = select_primary_actor(frames)
    _write_jsonl(target_path, targets)
    _write_json(quality_path, quality.as_dict())
    return quality


def generate(args: argparse.Namespace) -> Path:
    if min(args.video_batch_size, args.decode_workers, args.actor_workers) <= 0:
        raise ValueError("Batch and worker settings must be positive")
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    records = _records(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[ManifestVideo, BatchVideoTask, Path]] = []
    for record in records:
        track_path = _artifact_path(
            args.output_root / "tracks", record.path, ".jsonl"
        )
        quality_path = _artifact_path(
            args.output_root / "track_quality", record.path, ".json"
        )
        if track_path.exists() and quality_path.exists() and not args.overwrite:
            continue
        tasks.append(
            (
                record,
                BatchVideoTask(
                    video_path=record.video_path,
                    output_jsonl=track_path,
                    relative_path=record.path,
                    annotations=None,
                ),
                quality_path,
            )
        )

    config = TeacherConfig(
        model=args.model,
        image_size=tuple(args.imgsz),
        confidence=args.confidence,
        iou=args.iou,
        tracker=args.tracker,
        device=args.device,
        quantize=args.quantize,
        end2end=args.end2end,
    )
    started = time.perf_counter()
    processed_tracks = 0
    if tasks:
        from ultralytics import YOLO

        model = YOLO(args.model)
        for start in range(0, len(tasks), args.video_batch_size):
            chunk = tasks[start : start + args.video_batch_size]
            qualities = track_video_batch(
                model,
                [item[1] for item in chunk],
                config=config,
                decode_workers=args.decode_workers,
            )
            for (_, _, quality_path), quality in zip(
                chunk, qualities, strict=True
            ):
                _write_json(quality_path, quality.as_dict())
                processed_tracks += 1
            print(
                f"tracks={min(start + len(chunk), len(tasks))}/{len(tasks)}",
                flush=True,
            )

    processed_actors = 0

    def actor(record: ManifestVideo) -> ActorSelectionQuality | None:
        return _build_actor_boxes(
            record, args.output_root, overwrite=args.overwrite
        )

    with ThreadPoolExecutor(max_workers=args.actor_workers) as executor:
        for index, quality in enumerate(executor.map(actor, records), start=1):
            processed_actors += int(quality is not None)
            if index % 20 == 0 or index == len(records):
                print(f"actor_boxes={index}/{len(records)}", flush=True)

    successful = sum(
        _artifact_path(args.output_root / "actor_boxes", row.path, ".jsonl").is_file()
        for row in records
    )
    summary = {
        "domain": args.domain,
        "splits": list(args.splits),
        "selected_videos": len(records),
        "successful_videos": successful,
        "failed_videos": len(records) - successful,
        "processed_tracks_this_run": processed_tracks,
        "processed_actor_paths_this_run": processed_actors,
        "duration_s": time.perf_counter() - started,
        "label_blind": True,
    }
    summary_path = args.output_root / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/multidomain_cauca60/split_manifest.csv"),
    )
    parser.add_argument("--domain", default="caucafall")
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/external_pseudo_labels/caucafall/cauca60_yolo26x_nms"
        ),
    )
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--imgsz", type=int, nargs=2, default=(544, 960))
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--quantize", type=int, choices=(16, 32), default=16)
    parser.add_argument("--end2end", action="store_true")
    parser.add_argument("--video-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--actor-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
