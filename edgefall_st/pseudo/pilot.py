"""Build and run a stratified YOLO26 tracking pilot on OmniFall."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from edgefall_st.data.annotations import clean_annotations, load_annotations
from edgefall_st.pseudo.quality import (
    load_quality,
    summarize_quality,
    write_quality,
)
from edgefall_st.pseudo.sampling import (
    load_split_manifest,
    read_pilot_manifest,
    select_pilot_videos,
    write_pilot_manifest,
)


@dataclass(frozen=True)
class _WorkItem:
    order: int
    path: str
    task: object
    quality_path: Path
    error_path: Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _artifact_path(root: Path, relative_video_path: str, suffix: str) -> Path:
    relative = Path(*relative_video_path.split("/"))
    return root / relative.parent / f"{relative.name}{suffix}"


def ensure_manifest(args: argparse.Namespace) -> Path:
    manifest_path = args.output_dir / "pilot_manifest.csv"
    if manifest_path.exists() and not args.reselect:
        return manifest_path
    candidates = load_split_manifest(args.split_manifest, split=args.split)
    selected = select_pilot_videos(
        candidates,
        sample_size=args.sample_size,
        seed=args.seed,
        minimum_per_label=args.minimum_per_label,
    )
    write_pilot_manifest(selected, manifest_path)
    return manifest_path


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.video_batch_size <= 0:
        raise ValueError("video_batch_size must be positive")
    if args.decode_workers <= 0:
        raise ValueError("decode_workers must be positive")

    manifest_path = ensure_manifest(args)
    selected = read_pilot_manifest(manifest_path)
    if args.select_only:
        return {
            "manifest": str(manifest_path.resolve()),
            "selected_videos": len(selected),
        }

    from ultralytics import YOLO
    from edgefall_st.pseudo.batched_tracker import (
        BatchVideoTask,
        track_video_batch,
    )
    from edgefall_st.pseudo.yolo26_tracker import TeacherConfig

    segments, _ = load_annotations(args.dataset_root / "of-syn.csv")
    videos, _ = clean_annotations(segments)
    annotations = {video.path: video for video in videos}
    model = None
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
    _write_json(
        args.output_dir / "run_config.json",
        {
            "model": args.model,
            "image_size": list(args.imgsz),
            "confidence": args.confidence,
            "iou": args.iou,
            "tracker": args.tracker,
            "device": args.device,
            "quantize": args.quantize,
            "end2end": args.end2end,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "split": args.split,
            "video_batch_size": args.video_batch_size,
            "decode_workers": args.decode_workers,
            "inference_mode": "batched_predict_independent_bytetrack",
        },
    )

    processing = selected[: args.limit] if args.limit else selected
    work_items: list[_WorkItem] = []
    for order, video in enumerate(processing):
        video_file = args.dataset_root / f"{video.path}.mp4"
        tracks_path = _artifact_path(
            args.output_dir / "tracks",
            video.path,
            ".jsonl",
        )
        quality_path = _artifact_path(
            args.output_dir / "quality",
            video.path,
            ".json",
        )
        error_path = _artifact_path(
            args.output_dir / "errors",
            video.path,
            ".json",
        )
        if quality_path.exists() and tracks_path.exists() and not args.overwrite:
            error_path.unlink(missing_ok=True)
            print(f"[{order + 1}/{len(processing)}] skip {video.path}")
            continue
        visualization_path = None
        if order < args.visualizations:
            visualization_path = _artifact_path(
                args.output_dir / "visualizations",
                video.path,
                ".mp4",
            )
        work_items.append(
            _WorkItem(
                order=order,
                path=video.path,
                task=BatchVideoTask(
                    video_path=video_file,
                    output_jsonl=tracks_path,
                    relative_path=video.path,
                    annotations=annotations[video.path],
                    visualization_path=visualization_path,
                ),
                quality_path=quality_path,
                error_path=error_path,
            )
        )

    if work_items:
        model = YOLO(args.model)
        processing_started_at = time.perf_counter()
        cuda_available = False
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
    else:
        processing_started_at = None
        cuda_available = False
        torch = None

    def process_batch(items: list[_WorkItem]) -> None:
        print(
            f"[batch {items[0].order + 1}-{items[-1].order + 1}/"
            f"{len(processing)}] track "
            + ", ".join(item.path for item in items)
        )
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
                    f"  batch ERROR {type(error).__name__}: {error}; "
                    "retrying videos individually"
                )
                for item in items:
                    process_batch([item])
                return
            item = items[0]
            _write_json(
                item.error_path,
                {
                    "path": item.path,
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            if args.fail_fast:
                raise
            print(f"  ERROR {item.path}: {type(error).__name__}: {error}")
            return

        for item, quality in zip(items, qualities, strict=True):
            write_quality(quality, item.quality_path)
            item.error_path.unlink(missing_ok=True)
            print(
                f"  {item.path}: coverage={quality.detection_coverage:.3f} "
                f"dominant={quality.dominant_track_coverage:.3f} "
                f"tracks={quality.unique_track_ids} "
                f"elapsed={quality.elapsed_s:.2f}s"
            )

    for start in range(0, len(work_items), args.video_batch_size):
        process_batch(work_items[start : start + args.video_batch_size])

    processing_wall_time_s = None
    peak_cuda_memory_mb = None
    if processing_started_at is not None:
        if cuda_available and torch is not None:
            torch.cuda.synchronize()
            peak_cuda_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )
        processing_wall_time_s = time.perf_counter() - processing_started_at

    quality_files = sorted((args.output_dir / "quality").rglob("*.json"))
    qualities = [load_quality(path) for path in quality_files]
    summary = summarize_quality(qualities)
    summary.update(
        {
            "manifest": str(manifest_path.resolve()),
            "selected_videos": len(selected),
            "model": args.model,
            "processed_in_this_run": len(work_items),
            "video_batch_size": args.video_batch_size,
            "decode_workers": args.decode_workers,
        }
    )
    if processing_wall_time_s is not None:
        summary["processing_wall_time_s"] = processing_wall_time_s
        summary["throughput_videos_per_s"] = (
            len(work_items) / processing_wall_time_s
        )
    if peak_cuda_memory_mb is not None:
        summary["peak_cuda_memory_mb"] = peak_cuda_memory_mb
    _write_json(args.output_dir / "quality_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/omnifall_v1/split_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pseudo_labels/pilot_yolo26x_nms"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--minimum-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(544, 960),
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--quantize", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--end2end",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--visualizations", type=int, default=30)
    parser.add_argument("--video-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
