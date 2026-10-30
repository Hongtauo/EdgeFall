"""Extract frozen YOLO global/ROI features into a V3.4 cache."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from edgefall_st.data.dataset import (
    OmniFallWindowDataset,
    _letterbox_parameters,
    omnifall_indexed_collate_fn,
)
from edgefall_st.features.adapter import (
    FrozenFeatureBatch,
    UltralyticsYOLOFeatureAdapter,
)
from edgefall_st.features.cache import FrozenFeatureCacheWriter
from edgefall_st.training.sampler import VideoGroupedBatchSampler


DEFAULT_HISTORY_OFFSETS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
)
DEFAULT_VISUAL_OFFSETS = (0, 2, 4, 7, 8, 16, 24, 32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("Dataset/omnifall/of-syn")
    )
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--tracks-root", type=Path)
    parser.add_argument("--actor-targets-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("yolo26x.pt"))
    parser.add_argument("--input-height", type=int, default=544)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--geometry-input-height", type=int, default=288)
    parser.add_argument("--geometry-input-width", type=int, default=512)
    parser.add_argument("--roi-output-size", type=int, default=3)
    parser.add_argument("--roi-context-scale", type=float, default=1.25)
    parser.add_argument(
        "--global-levels", nargs="+", default=("p4", "p5")
    )
    parser.add_argument("--roi-levels", nargs="+", default=("p3", "p4"))
    parser.add_argument(
        "--history-offsets", nargs="+", type=int, default=DEFAULT_HISTORY_OFFSETS
    )
    parser.add_argument(
        "--visual-offsets", nargs="+", type=int, default=DEFAULT_VISUAL_OFFSETS
    )
    parser.add_argument("--effective-fps", type=float, default=10.0)
    parser.add_argument("--windows-per-batch", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument(
        "--actor-top-k",
        type=int,
        default=1,
        help="Cache this many mutually exclusive actor paths per video",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    split = args.split
    args.windows = args.windows or Path(
        f"artifacts/omnifall_v2/windows_{split}.jsonl"
    )
    args.tracks_root = args.tracks_root or Path(
        f"artifacts/pseudo_labels/{split}_yolo26x_nms/tracks"
    )
    args.actor_targets_root = args.actor_targets_root or Path(
        f"artifacts/actor_targets_v1/{split}/targets"
    )
    args.output_root = args.output_root or Path(
        f"artifacts/frozen_features/yolo26x/{split}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "input_height",
        "input_width",
        "geometry_input_height",
        "geometry_input_width",
        "roi_output_size",
        "windows_per_batch",
        "feature_batch_size",
        "actor_top_k",
        "log_interval",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.max_batches < 0:
        raise ValueError("workers/max-batches cannot be negative")
    if args.input_height % 32 or args.input_width % 32:
        raise ValueError("YOLO input dimensions must be divisible by 32")
    if args.output_root.exists():
        raise FileExistsError(
            f"Output cache already exists: {args.output_root}"
        )
    if args.effective_fps <= 0 or args.roi_context_scale < 1.0:
        raise ValueError("Invalid fps or ROI context scale")
    history = tuple(int(value) for value in args.history_offsets)
    visual = tuple(int(value) for value in args.visual_offsets)
    if not history or history[0] != 0 or tuple(sorted(set(history))) != history:
        raise ValueError("history-offsets must be unique, increasing, and start at 0")
    if not visual or visual[0] != 0 or tuple(sorted(set(visual))) != visual:
        raise ValueError("visual-offsets must be unique, increasing, and start at 0")
    if not set(visual).issubset(history):
        raise ValueError("visual-offsets must be a subset of history-offsets")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remap_letterboxed_boxes(
    boxes: Tensor,
    source_sizes: Sequence[tuple[int, int]],
    *,
    source_input_size: tuple[int, int],
    target_input_size: tuple[int, int],
) -> Tensor:
    """Map normalized boxes between two letterboxes of the same source frame."""

    if boxes.ndim < 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B, ..., 4]")
    if len(source_sizes) != boxes.shape[0]:
        raise ValueError("source_sizes must contain one value per batch item")
    result = boxes.clone().float()
    source_input_h, source_input_w = source_input_size
    target_h, target_w = target_input_size
    for batch_index, (source_h, source_w) in enumerate(source_sizes):
        old_scale, old_left, old_top = _letterbox_parameters(
            (int(source_h), int(source_w)), source_input_size
        )
        new_scale, new_left, new_top = _letterbox_parameters(
            (int(source_h), int(source_w)), target_input_size
        )
        original_shape = result[batch_index].shape
        values = result[batch_index].reshape(-1, 4)
        x1 = (values[:, 0] * source_input_w - old_left) / old_scale
        y1 = (values[:, 1] * source_input_h - old_top) / old_scale
        x2 = (values[:, 2] * source_input_w - old_left) / old_scale
        y2 = (values[:, 3] * source_input_h - old_top) / old_scale
        remapped = torch.stack(
            (
                (x1 * new_scale + new_left) / target_w,
                (y1 * new_scale + new_top) / target_h,
                (x2 * new_scale + new_left) / target_w,
                (y2 * new_scale + new_top) / target_h,
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        result[batch_index] = remapped.reshape(original_shape)
    return result


def _extract_chunks(
    adapter: UltralyticsYOLOFeatureAdapter,
    images: Tensor,
    boxes: Tensor,
    actor_mask: Tensor,
    *,
    chunk_size: int,
    device: torch.device,
    amp: bool,
) -> FrozenFeatureBatch:
    global_parts: list[Tensor] = []
    roi_parts: list[Tensor] = []
    for start in range(0, images.shape[0], chunk_size):
        stop = min(images.shape[0], start + chunk_size)
        image_chunk = images[start:stop].to(device, non_blocking=True).float().div_(255.0)
        box_chunk = boxes[start:stop].to(device, non_blocking=True)
        mask_chunk = actor_mask[start:stop].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = adapter(image_chunk, box_chunk, mask_chunk)
        global_parts.append(output.global_features.float().cpu())
        roi_parts.append(output.roi_features.float().cpu())
    return FrozenFeatureBatch(
        global_features=torch.cat(global_parts),
        roi_features=torch.cat(roi_parts),
        actor_valid_mask=actor_mask.cpu(),
    )


def _target_scalar(target: dict[str, Tensor], name: str) -> float | int:
    value = target[name].detach().cpu().item()
    return int(value) if name in {"action_label", "stage_label"} else float(value)


def _records(
    targets: Sequence[dict[str, Tensor]],
    metadata: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target, item in zip(targets, metadata, strict=True):
        result.append(
            {
                "path": str(item["path"]),
                "video_has_fall_event": bool(item["video_has_fall_event"]),
                "event_target": bool(item["event_target"]),
                "window_index": int(item["window_index"]),
                "end_frame": int(item["end_frame"]),
                "end_time_s": float(item["end_time_s"]),
                "window_action_label": int(item["window_action_label"]),
                "action_label": _target_scalar(target, "action_label"),
                "stage_label": _target_scalar(target, "stage_label"),
                "onset_target": _target_scalar(target, "onset_target"),
                "action_loss_weight": _target_scalar(
                    target, "action_loss_weight"
                ),
                "history_frame_indices": [
                    int(value) for value in item["history_frame_indices"]
                ],
                "source_size": [int(value) for value in item["source_size"]],
            }
        )
    return result


def _unique_frame_indices(
    history_indices: Tensor,
    metadata: Sequence[dict[str, Any]],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for sample_index, item in enumerate(metadata):
        frame_indices = item["history_frame_indices"]
        for slot, frame_value in enumerate(frame_indices):
            unique_index = int(history_indices[sample_index, slot])
            frame_index = int(frame_value)
            if unique_index < 0:
                if frame_index >= 0:
                    raise ValueError("Invalid unique/history frame mapping")
                continue
            previous = result.get(unique_index)
            if previous is not None and previous != frame_index:
                raise ValueError("One unique image maps to two source frames")
            result[unique_index] = frame_index
    return result


def _same_tensor(first: Tensor, second: Tensor) -> bool:
    return first.shape == second.shape and torch.allclose(
        first, second, atol=1e-6, rtol=1e-5
    )


def run_extraction(args: argparse.Namespace) -> Path:
    _resolve_paths(args)
    _validate_args(args)
    device = _device(args.device)
    history_offsets = tuple(int(value) for value in args.history_offsets)
    visual_offsets = tuple(int(value) for value in args.visual_offsets)
    slot_by_offset = {value: index for index, value in enumerate(history_offsets)}
    visual_slots = torch.tensor(
        [slot_by_offset[value] for value in visual_offsets], dtype=torch.long
    )

    dataset = OmniFallWindowDataset(
        dataset_root=args.dataset_root,
        windows_path=args.windows,
        tracks_root=args.tracks_root,
        actor_targets_root=args.actor_targets_root,
        input_size=(args.input_height, args.input_width),
        effective_fps=args.effective_fps,
        history_offsets=history_offsets,
        return_uint8=True,
        actor_top_k=args.actor_top_k,
        video_frame_cache_size=1,
    )
    sampler = VideoGroupedBatchSampler(
        dataset.windows,
        batch_size=args.windows_per_batch,
        windows_per_video=args.windows_per_batch,
        shuffle=False,
        drop_last=False,
    )
    loader_options: dict[str, Any] = {}
    if args.workers > 0:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        collate_fn=omnifall_indexed_collate_fn,
        pin_memory=device.type == "cuda",
        **loader_options,
    )
    adapter = UltralyticsYOLOFeatureAdapter(
        args.weights,
        global_levels=args.global_levels,
        roi_levels=args.roi_levels,
        roi_output_size=args.roi_output_size,
        roi_context_scale=args.roi_context_scale,
    ).to(device)
    adapter.eval()
    writer: FrozenFeatureCacheWriter | None = None
    current_path: str | None = None
    current_source_size: tuple[int, int] | None = None
    geometry_by_frame: dict[int, tuple[Tensor, Tensor]] = {}
    visual_by_frame: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
    window_records: list[dict[str, Any]] = []
    processed_windows = 0

    def flush_video() -> None:
        nonlocal current_path, current_source_size
        if current_path is None:
            return
        if writer is None or current_source_size is None:
            raise RuntimeError("Feature writer was not initialized")
        geometry_frames = sorted(geometry_by_frame)
        visual_frames = sorted(visual_by_frame)
        writer.write_video(
            current_path,
            frame_indices=np.asarray(geometry_frames, dtype=np.int64),
            visual_frame_indices=np.asarray(visual_frames, dtype=np.int64),
            actor_boxes=np.stack(
                [geometry_by_frame[frame][0].numpy() for frame in geometry_frames]
            ),
            actor_valid_mask=np.asarray(
                [
                    geometry_by_frame[frame][1].numpy()
                    for frame in geometry_frames
                ],
                dtype=np.float32,
            ),
            visual_actor_boxes=np.stack(
                [visual_by_frame[frame][0].numpy() for frame in visual_frames]
            ),
            global_features=np.stack(
                [visual_by_frame[frame][1].numpy() for frame in visual_frames]
            ).astype(np.float16),
            roi_features=np.stack(
                [visual_by_frame[frame][2].numpy() for frame in visual_frames]
            ).astype(np.float16),
            source_size=current_source_size,
            records=window_records,
        )
        geometry_by_frame.clear()
        visual_by_frame.clear()
        window_records.clear()
    start_time = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                batch_paths = {str(item["path"]) for item in batch["metadata"]}
                if len(batch_paths) != 1:
                    raise RuntimeError(
                        "Extraction batches must contain exactly one video"
                    )
                batch_path = next(iter(batch_paths))
                batch_source_sizes = {
                    tuple(int(value) for value in item["source_size"])
                    for item in batch["metadata"]
                }
                if len(batch_source_sizes) != 1:
                    raise ValueError("A video's source dimensions changed")
                batch_source_size = next(iter(batch_source_sizes))
                if current_path is not None and batch_path != current_path:
                    flush_video()
                if current_path != batch_path:
                    current_path = batch_path
                    current_source_size = batch_source_size

                history_indices = batch["history_indices"]
                frame_by_unique = _unique_frame_indices(
                    history_indices, batch["metadata"]
                )
                selected_indices = history_indices.index_select(1, visual_slots)
                referenced = torch.unique(selected_indices[selected_indices.ge(0)])
                if referenced.numel() == 0:
                    raise RuntimeError("Batch has no valid visual frames")
                new_referenced = torch.tensor(
                    [
                        int(unique_index)
                        for unique_index in referenced.tolist()
                        if frame_by_unique[int(unique_index)]
                        not in visual_by_frame
                    ],
                    dtype=torch.long,
                )
                if new_referenced.numel():
                    features = _extract_chunks(
                        adapter,
                        batch["images"].index_select(0, new_referenced),
                        batch["actor_boxes"].index_select(0, new_referenced),
                        batch["actor_valid_mask"].index_select(
                            0, new_referenced
                        ),
                        chunk_size=args.feature_batch_size,
                        device=device,
                        amp=args.amp,
                    )
                    for feature_row, unique_index in enumerate(
                        new_referenced.tolist()
                    ):
                        frame_index = frame_by_unique[int(unique_index)]
                        visual_by_frame[frame_index] = (
                            batch["actor_boxes"][unique_index].clone(),
                            features.global_features[feature_row].clone(),
                            features.roi_features[feature_row].clone(),
                        )

                safe_history = history_indices.clamp_min(0)
                geometry_boxes = batch["actor_boxes"][safe_history]
                geometry_actor_mask = batch["actor_valid_mask"][safe_history]
                history_valid = history_indices.ge(0) * batch["valid_mask"]
                if geometry_actor_mask.ndim == 3:
                    history_valid = history_valid[..., None]
                geometry_actor_mask = geometry_actor_mask * history_valid
                geometry_boxes = _remap_letterboxed_boxes(
                    geometry_boxes,
                    [tuple(item["source_size"]) for item in batch["metadata"]],
                    source_input_size=(args.input_height, args.input_width),
                    target_input_size=(
                        args.geometry_input_height,
                        args.geometry_input_width,
                    ),
                )
                geometry_boxes *= geometry_actor_mask[..., None]

                for sample_index, item in enumerate(batch["metadata"]):
                    for slot, frame_value in enumerate(
                        item["history_frame_indices"]
                    ):
                        frame_index = int(frame_value)
                        if frame_index < 0:
                            continue
                        value = (
                            geometry_boxes[sample_index, slot].clone(),
                            geometry_actor_mask[sample_index, slot].clone(),
                        )
                        previous = geometry_by_frame.get(frame_index)
                        if previous is not None and (
                            not _same_tensor(previous[1], value[1])
                            or not _same_tensor(previous[0], value[0])
                        ):
                            raise ValueError(
                                "Repeated geometry frames are inconsistent"
                            )
                        geometry_by_frame[frame_index] = value

                if writer is None:
                    writer = FrozenFeatureCacheWriter(
                        args.output_root,
                        metadata={
                            "split": args.split,
                            "history_offsets": list(history_offsets),
                            "visual_offsets": list(visual_offsets),
                            "visual_input_size": [
                                args.input_height,
                                args.input_width,
                            ],
                            "geometry_input_size": [
                                args.geometry_input_height,
                                args.geometry_input_width,
                            ],
                            "effective_fps": args.effective_fps,
                            "weights_sha256": _sha256(args.weights),
                            "windows_sha256": _sha256(args.windows),
                            "data_sources": {
                                "dataset_root": str(args.dataset_root),
                                "windows": str(args.windows),
                                "tracks_root": str(args.tracks_root),
                                "actor_targets_root": str(
                                    args.actor_targets_root
                                ),
                            },
                            "feature_adapter": adapter.feature_spec(),
                            "actor_top_k": args.actor_top_k,
                        },
                    )
                window_records.extend(
                    _records(batch["targets"], batch["metadata"])
                )
                processed_windows += len(batch["targets"])
                if (batch_index + 1) % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"batch={batch_index + 1}/{len(loader)} "
                        f"windows={processed_windows} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
        if writer is None:
            raise RuntimeError("No feature batches were extracted")
        flush_video()
        manifest = writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        adapter.close()
    print(f"feature_cache={manifest}", flush=True)
    return manifest


def main() -> None:
    run_extraction(build_parser().parse_args())


if __name__ == "__main__":
    main()
