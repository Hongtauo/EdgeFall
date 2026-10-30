"""Generate YOLO26x tracks and label-blind primary-actor boxes for external videos."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
from external_eval.fall_video_dataset.common import VideoRecord, discover_records


@dataclass(frozen=True)
class _TrackWorkItem:
    order: int
    record: VideoRecord
    task: BatchVideoTask
    quality_path: Path
    error_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/fall-video-dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/external_pseudo_labels/fall_video_dataset/yolo26x_nms"
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
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _artifact_path(root: Path, sample_id: str, suffix: str) -> Path:
    relative = Path(*sample_id.split("/"))
    return root / relative.parent / f"{relative.name}{suffix}"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _select_records(
    records: Sequence[VideoRecord],
    *,
    limit_per_class: int,
    seed: int,
) -> list[VideoRecord]:
    if limit_per_class <= 0:
        return list(records)
    selected: list[VideoRecord] = []
    randomizer = random.Random(seed)
    for label_name in ("Fall", "No_Fall"):
        candidates = [record for record in records if record.label_name == label_name]
        randomizer.shuffle(candidates)
        selected.extend(candidates[:limit_per_class])
    return sorted(selected, key=lambda record: record.sample_id.lower())


def _run_config(args: argparse.Namespace, selected: Sequence[VideoRecord]) -> dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "model": str(Path(args.model).resolve()),
        "image_size": list(args.imgsz),
        "confidence": args.confidence,
        "iou": args.iou,
        "tracker": args.tracker,
        "device": args.device,
        "quantize": args.quantize,
        "end2end": args.end2end,
        "video_batch_size": args.video_batch_size,
        "decode_workers": args.decode_workers,
        "actor_workers": args.actor_workers,
        "limit_per_class": args.limit_per_class,
        "seed": args.seed,
        "selected_videos": len(selected),
        "inference_mode": "batched_predict_independent_bytetrack",
        "actor_selection": "label_blind_temporally_coherent_primary_path",
    }


def _check_config(path: Path, expected: dict[str, object], overwrite: bool) -> None:
    if not path.exists() or overwrite:
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != expected:
        raise ValueError(
            f"Configuration changed for {path.parent}. Use a new output root "
            "or pass --overwrite."
        )


def _write_manifest(
    path: Path,
    records: Sequence[VideoRecord],
    output_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ("sample_id", "label_name", "video_path", "actor_boxes_path")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "label_name": record.label_name,
                    "video_path": str(record.video_path.resolve()),
                    "actor_boxes_path": str(
                        _artifact_path(
                            output_root / "actor_boxes",
                            record.sample_id,
                            ".jsonl",
                        ).resolve()
                    ),
                }
            )
    temporary.replace(path)


def _actor_quality_path(root: Path, sample_id: str) -> Path:
    return _artifact_path(root / "actor_quality", sample_id, ".json")


def _build_actor_boxes(
    record: VideoRecord,
    output_root: Path,
    *,
    overwrite: bool,
) -> ActorSelectionQuality | None:
    track_path = _artifact_path(output_root / "tracks", record.sample_id, ".jsonl")
    target_path = _artifact_path(
        output_root / "actor_boxes", record.sample_id, ".jsonl"
    )
    quality_path = _actor_quality_path(output_root, record.sample_id)
    if target_path.exists() and quality_path.exists() and not overwrite:
        return None
    if not track_path.exists():
        return None
    frames = load_track_frames(track_path)
    targets, quality = select_primary_actor(frames)
    _write_jsonl_atomic(target_path, targets)
    _write_json_atomic(quality_path, quality.as_dict())
    return quality


def generate(args: argparse.Namespace) -> Path:
    if args.video_batch_size <= 0 or args.decode_workers <= 0 or args.actor_workers <= 0:
        raise ValueError("worker and batch settings must be positive")
    if args.limit_per_class < 0:
        raise ValueError("limit-per-class must be non-negative")
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)

    records = _select_records(
        discover_records(args.dataset_root),
        limit_per_class=args.limit_per_class,
        seed=args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = _run_config(args, records)
    config_path = args.output_root / "run_config.json"
    _check_config(config_path, config, args.overwrite)
    _write_json_atomic(config_path, config)
    _write_manifest(args.output_root / "manifest.csv", records, args.output_root)

    work_items: list[_TrackWorkItem] = []
    for order, record in enumerate(records):
        track_path = _artifact_path(
            args.output_root / "tracks", record.sample_id, ".jsonl"
        )
        quality_path = _artifact_path(
            args.output_root / "track_quality", record.sample_id, ".json"
        )
        if track_path.exists() and quality_path.exists() and not args.overwrite:
            continue
        work_items.append(
            _TrackWorkItem(
                order=order,
                record=record,
                task=BatchVideoTask(
                    video_path=record.video_path,
                    output_jsonl=track_path,
                    relative_path=record.sample_id,
                    annotations=None,
                ),
                quality_path=quality_path,
                error_path=_artifact_path(
                    args.output_root / "errors", record.sample_id, ".json"
                ),
            )
        )

    teacher_config = TeacherConfig(
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
    if work_items:
        from ultralytics import YOLO

        model = YOLO(args.model)

        def process_batch(items: list[_TrackWorkItem]) -> None:
            nonlocal processed_tracks
            try:
                qualities = track_video_batch(
                    model,
                    [item.task for item in items],
                    config=teacher_config,
                    decode_workers=args.decode_workers,
                )
            except Exception as error:
                if len(items) > 1:
                    print(
                        f"batch {items[0].order + 1}-{items[-1].order + 1} "
                        f"failed ({type(error).__name__}); retrying individually",
                        flush=True,
                    )
                    for item in items:
                        process_batch([item])
                    return
                item = items[0]
                _write_json_atomic(
                    item.error_path,
                    {
                        "sample_id": item.record.sample_id,
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                )
                print(
                    f"track ERROR {item.record.sample_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if args.fail_fast:
                    raise
                return
            for item, quality in zip(items, qualities, strict=True):
                _write_json_atomic(item.quality_path, quality.as_dict())
                item.error_path.unlink(missing_ok=True)
                processed_tracks += 1
            elapsed = time.perf_counter() - started
            completed = items[-1].order + 1
            print(
                f"tracks={completed}/{len(records)} "
                f"new={processed_tracks} videos_per_s="
                f"{processed_tracks / max(elapsed, 1e-6):.2f}",
                flush=True,
            )

        for start in range(0, len(work_items), args.video_batch_size):
            process_batch(work_items[start : start + args.video_batch_size])

    actor_started = time.perf_counter()
    processed_actors = 0

    def process_actor(record: VideoRecord) -> ActorSelectionQuality | None:
        return _build_actor_boxes(
            record,
            args.output_root,
            overwrite=args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=args.actor_workers) as executor:
        for index, quality in enumerate(executor.map(process_actor, records), start=1):
            processed_actors += int(quality is not None)
            if index % 100 == 0 or index == len(records):
                print(
                    f"actor_boxes={index}/{len(records)} new={processed_actors}",
                    flush=True,
                )

    actor_qualities = []
    successful = 0
    for record in records:
        quality_path = _actor_quality_path(args.output_root, record.sample_id)
        if not quality_path.exists():
            continue
        successful += 1
        actor_qualities.append(json.loads(quality_path.read_text(encoding="utf-8")))
    summary = {
        "selected_videos": len(records),
        "successful_videos": successful,
        "failed_videos": len(records) - successful,
        "processed_tracks_this_run": processed_tracks,
        "processed_actor_paths_this_run": processed_actors,
        "tracking_and_selection_duration_s": time.perf_counter() - started,
        "actor_selection_duration_s": time.perf_counter() - actor_started,
        "mean_actor_coverage": (
            sum(float(value["actor_coverage"]) for value in actor_qualities)
            / max(len(actor_qualities), 1)
        ),
        "label_blind": True,
    }
    summary_path = args.output_root / "summary.json"
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    generate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
