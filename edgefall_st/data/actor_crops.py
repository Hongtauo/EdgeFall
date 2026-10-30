"""Precompute temporally aligned actor crops for bbox-classifier training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from edgefall_st.data.dataset import (
    _letterbox_image_uint8,
    _read_jsonl,
    _relative_sample_path,
    _transform_normalized_box,
    _with_suffix,
)
from edgefall_st.models.bbox_classifier import ActorCropper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    parser.add_argument("--actor-targets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=128)
    parser.add_argument("--crop-width", type=int, default=128)
    parser.add_argument("--crop-context", type=float, default=1.25)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--video-suffix", default=".mp4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    dimensions = (
        args.input_height,
        args.input_width,
        args.crop_height,
        args.crop_width,
        args.batch_size,
        args.log_interval,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("Dimensions, batch-size, and log-interval must be positive")
    if args.crop_context < 1.0:
        raise ValueError("crop-context must be at least 1")
    if args.limit < 0:
        raise ValueError("limit cannot be negative")
    if not args.video_suffix.startswith("."):
        raise ValueError("video-suffix must start with '.'")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _output_path(
    actor_path: Path,
    actor_targets_root: Path,
    output_root: Path,
) -> Path:
    return (output_root / actor_path.relative_to(actor_targets_root)).with_suffix(
        ".npy"
    )


def _valid_existing_cache(
    path: Path,
    *,
    frame_count: int,
    crop_size: tuple[int, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = values.dtype == np.uint8 and values.shape == (
            frame_count,
            3,
            *crop_size,
        )
        del values
        return bool(valid)
    except (OSError, ValueError):
        return False


def _decode_letterboxed_video(
    video_path: Path,
    actors: Sequence[dict[str, Any]],
    *,
    input_size: tuple[int, int],
) -> tuple[list[torch.Tensor], list[list[float] | None]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    images: list[torch.Tensor] = []
    boxes: list[list[float] | None] = []
    try:
        for frame_index, actor in enumerate(actors):
            success, image = capture.read()
            if not success or image is None:
                raise RuntimeError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            source_h, source_w = image.shape[:2]
            image_uint8, scale, pad_left, pad_top = _letterbox_image_uint8(
                image,
                input_size,
                fill_value=114,
            )
            box = _transform_normalized_box(
                actor.get("actor_bbox_xyxy_norm"),
                source_size=(source_h, source_w),
                input_size=input_size,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
            )
            images.append(image_uint8)
            boxes.append(box)
    finally:
        capture.release()
    return images, boxes


def _crop_video(
    images: Sequence[torch.Tensor],
    boxes: Sequence[list[float] | None],
    *,
    cropper: ActorCropper,
    crop_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = np.zeros(
        (len(images), 3, *crop_size),
        dtype=np.uint8,
    )
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            image_batch = torch.stack(images[start:end]).to(
                device=device,
                dtype=torch.float32,
            ).div_(255.0)
            valid = torch.tensor(
                [box is not None for box in boxes[start:end]],
                device=device,
                dtype=torch.bool,
            )
            box_batch = torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0] if box is None else box
                    for box in boxes[start:end]
                ],
                device=device,
                dtype=torch.float32,
            )
            crops = cropper(image_batch, box_batch)
            crops = crops.mul(255.0).round().clamp_(0.0, 255.0).byte()
            crops[~valid] = 0
            output[start:end] = crops.cpu().numpy()
    return output


def _atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    device = _resolve_device(args.device)
    input_size = (args.input_height, args.input_width)
    crop_size = (args.crop_height, args.crop_width)
    actor_paths = sorted(args.actor_targets_root.rglob("*.jsonl"))
    if args.limit:
        actor_paths = actor_paths[: args.limit]
    if not actor_paths:
        raise ValueError(f"No actor targets found in {args.actor_targets_root}")

    cropper = ActorCropper(
        output_size=crop_size,
        context_scale=args.crop_context,
    ).to(device).eval()
    started = time.perf_counter()
    processed = 0
    skipped = 0
    total_bytes = 0
    for video_index, actor_path in enumerate(actor_paths, start=1):
        actors = _read_jsonl(actor_path)
        if not actors:
            raise ValueError(f"No actor rows found in {actor_path}")
        sample_path = str(actors[0].get("path", ""))
        relative = _relative_sample_path(sample_path)
        video_path = args.dataset_root / _with_suffix(
            relative,
            args.video_suffix,
        )
        output_path = _output_path(
            actor_path,
            args.actor_targets_root,
            args.output_root,
        )
        if not args.overwrite and _valid_existing_cache(
            output_path,
            frame_count=len(actors),
            crop_size=crop_size,
        ):
            skipped += 1
            total_bytes += output_path.stat().st_size
        else:
            images, boxes = _decode_letterboxed_video(
                video_path,
                actors,
                input_size=input_size,
            )
            crops = _crop_video(
                images,
                boxes,
                cropper=cropper,
                crop_size=crop_size,
                batch_size=args.batch_size,
                device=device,
            )
            _atomic_save(output_path, crops)
            processed += 1
            total_bytes += output_path.stat().st_size
        if video_index % args.log_interval == 0 or video_index == len(actor_paths):
            elapsed = time.perf_counter() - started
            print(
                f"videos={video_index}/{len(actor_paths)} processed={processed} "
                f"skipped={skipped} videos_per_s={video_index / max(elapsed, 1e-9):.2f}",
                flush=True,
            )

    result = {
        "dataset_root": str(args.dataset_root.resolve()),
        "actor_targets_root": str(args.actor_targets_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "videos": len(actor_paths),
        "processed": processed,
        "skipped": skipped,
        "input_size": list(input_size),
        "crop_size": list(crop_size),
        "crop_context": args.crop_context,
        "dtype": "uint8",
        "layout": "NCHW",
        "bytes": total_bytes,
        "duration_s": time.perf_counter() - started,
        "device": str(device),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["summary"] = str(summary_path.resolve())
    return result


def main() -> None:
    result = generate(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
