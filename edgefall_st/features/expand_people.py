"""Expand a single-actor feature cache with label-blind extra actor ROIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from edgefall_st.data.dataset import (
    _additional_actor_paths,
    _letterbox_image_uint8,
    _letterbox_parameters,
    _read_jsonl,
    _relative_sample_path,
    _transform_normalized_box,
    _with_suffix,
)
from edgefall_st.features.adapter import UltralyticsYOLOFeatureAdapter
from edgefall_st.features.cache import FrozenFeatureCacheWriter


@dataclass(frozen=True)
class _PreparedVideo:
    sample_path: str
    records: list[dict[str, Any]]
    shard: dict[str, np.ndarray]
    source_size: tuple[int, int]
    geometry_boxes: np.ndarray
    geometry_valid: np.ndarray
    visual_boxes: np.ndarray
    visual_valid: np.ndarray
    visual_images: Tensor | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--actor-targets-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("yolo26s.pt"))
    parser.add_argument("--actor-top-k", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-videos", type=int, default=8)
    return parser


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


def _target_box(
    target: dict[str, Any],
    *,
    source_size: tuple[int, int],
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    scale, pad_left, pad_top = _letterbox_parameters(source_size, input_size)
    transformed = _transform_normalized_box(
        target.get("actor_bbox_xyxy_norm"),
        source_size=source_size,
        input_size=input_size,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    if transformed is None:
        return np.zeros(4, dtype=np.float32), 0.0
    return np.asarray(transformed, dtype=np.float32), 1.0


def _decode_frames(video_path: Path, frame_indices: Sequence[int]) -> list[np.ndarray]:
    requested = sorted(set(int(value) for value in frame_indices))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        requested_set = set(requested)
        for frame_index in range(requested[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            if frame_index in requested_set:
                decoded[frame_index] = frame
    finally:
        capture.release()
    return [decoded[int(value)] for value in frame_indices]


def _secondary_roi_features(
    adapter: UltralyticsYOLOFeatureAdapter,
    *,
    images: Tensor,
    boxes: np.ndarray,
    valid_mask: np.ndarray,
    roi_shape: tuple[int, int, int],
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    output = np.zeros((len(valid_mask), *roi_shape), dtype=np.float16)
    valid_rows = np.flatnonzero(valid_mask > 0)
    if valid_rows.size == 0:
        return output
    if images.shape[0] != len(valid_rows):
        raise ValueError("Prepared image and valid-row counts disagree")
    selected_boxes = torch.from_numpy(boxes[valid_rows]).float()
    for start in range(0, len(valid_rows), batch_size):
        stop = min(len(valid_rows), start + batch_size)
        image_batch = images[start:stop].to(device).float().div_(255.0)
        box_batch = selected_boxes[start:stop].to(device)
        mask_batch = torch.ones(stop - start, device=device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            features = adapter(image_batch, box_batch, mask_batch)
        output[valid_rows[start:stop]] = (
            features.roi_features.float().cpu().numpy().astype(np.float16)
        )
    return output


def _prepare_video(
    item: tuple[str, list[dict[str, Any]]],
    *,
    base_cache: Path,
    dataset_root: Path,
    tracks_root: Path,
    actor_targets_root: Path,
    visual_input: tuple[int, int],
    geometry_input: tuple[int, int],
) -> _PreparedVideo:
    cache_name, records = item
    sample_path = str(records[0]["path"])
    relative = _relative_sample_path(sample_path)
    annotation = _with_suffix(relative, ".jsonl")
    tracks = _read_jsonl(tracks_root / annotation)
    primary = _read_jsonl(actor_targets_root / annotation)
    secondary = _additional_actor_paths(tracks, primary, 1)[0]
    with np.load(
        base_cache / "videos" / cache_name,
        allow_pickle=False,
    ) as archive:
        shard = {key: np.array(archive[key], copy=True) for key in archive.files}
    source_size = tuple(int(value) for value in shard["source_size"])
    frame_indices = shard["frame_indices"]
    visual_frame_indices = shard["visual_frame_indices"]
    geometry_secondary = [
        _target_box(
            secondary[int(frame)],
            source_size=source_size,
            input_size=geometry_input,
        )
        for frame in frame_indices
    ]
    visual_secondary = [
        _target_box(
            secondary[int(frame)],
            source_size=source_size,
            input_size=visual_input,
        )
        for frame in visual_frame_indices
    ]
    geometry_boxes = np.stack([value[0] for value in geometry_secondary])
    geometry_valid = np.asarray(
        [value[1] for value in geometry_secondary], dtype=np.float32
    )
    visual_boxes = np.stack([value[0] for value in visual_secondary])
    visual_valid = np.asarray(
        [value[1] for value in visual_secondary], dtype=np.float32
    )
    valid_rows = np.flatnonzero(visual_valid > 0)
    visual_images: Tensor | None = None
    if valid_rows.size:
        selected_frames = visual_frame_indices[valid_rows]
        decoded = _decode_frames(
            dataset_root / _with_suffix(relative, ".mp4"),
            selected_frames.tolist(),
        )
        visual_images = torch.stack(
            [
                _letterbox_image_uint8(
                    frame, visual_input, fill_value=114
                )[0]
                for frame in decoded
            ]
        )
    return _PreparedVideo(
        sample_path=sample_path,
        records=records,
        shard=shard,
        source_size=source_size,
        geometry_boxes=geometry_boxes,
        geometry_valid=geometry_valid,
        visual_boxes=visual_boxes,
        visual_valid=visual_valid,
        visual_images=visual_images,
    )


def _bounded_prefetch(
    items: Sequence[tuple[str, list[dict[str, Any]]]],
    function: Callable[[tuple[str, list[dict[str, Any]]]], _PreparedVideo],
    *,
    workers: int,
    capacity: int,
) -> Iterator[_PreparedVideo]:
    if workers == 0:
        for item in items:
            yield function(item)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[_PreparedVideo]] = deque()
        iterator = iter(items)
        for _ in range(min(capacity, len(items))):
            pending.append(executor.submit(function, next(iterator)))
        while pending:
            yield pending.popleft().result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))


def _metadata(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "format",
        "complete",
        "sample_count",
        "video_count",
        "geometry_frame_count",
        "visual_frame_count",
        "arrays",
        "actor_top_k",
    }
    return {
        **{key: value for key, value in base.items() if key not in excluded},
        "actor_top_k": args.actor_top_k,
        "expanded_from_cache": str(args.base_cache),
        "extra_actor_selection": "label-blind-exclusive-viterbi",
    }


def run_expansion(args: argparse.Namespace) -> Path:
    if args.actor_top_k != 2:
        raise ValueError("The cache expander currently supports actor-top-k=2")
    if (
        min(
            args.feature_batch_size,
            args.log_interval,
            args.prefetch_videos,
        )
        <= 0
        or min(args.max_videos, args.workers) < 0
    ):
        raise ValueError("Batch and logging settings must be positive")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    base_manifest = json.loads(
        (args.base_cache / "manifest.json").read_text(encoding="utf-8")
    )
    if base_manifest["arrays"]["actor_boxes"] != [4]:
        raise ValueError("Base cache must contain exactly one actor")
    if _sha256(args.weights) != base_manifest["weights_sha256"]:
        raise ValueError("YOLO weights do not match the base cache")
    entries = _read_jsonl(args.base_cache / "index.jsonl")
    by_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_cache[str(entry["video_cache"])].append(entry)
    cache_groups = list(by_cache.items())
    if args.max_videos:
        cache_groups = cache_groups[: args.max_videos]

    device = _device(args.device)
    adapter_spec = base_manifest["feature_adapter"]
    adapter = UltralyticsYOLOFeatureAdapter(
        args.weights,
        global_levels=tuple(adapter_spec["global_levels"]),
        roi_levels=tuple(adapter_spec["roi_levels"]),
        roi_output_size=int(adapter_spec["roi_output_size"]),
        roi_context_scale=float(adapter_spec["roi_context_scale"]),
    ).to(device)
    adapter.eval()
    writer = FrozenFeatureCacheWriter(
        args.output_root,
        metadata=_metadata(base_manifest, args),
    )
    visual_input = tuple(int(value) for value in base_manifest["visual_input_size"])
    geometry_input = tuple(
        int(value) for value in base_manifest["geometry_input_size"]
    )
    roi_shape = tuple(int(value) for value in base_manifest["arrays"]["roi_features"])
    started = time.perf_counter()
    multi_person_videos = 0
    secondary_visual_frames = 0
    prepare_video = partial(
        _prepare_video,
        base_cache=args.base_cache,
        dataset_root=args.dataset_root,
        tracks_root=args.tracks_root,
        actor_targets_root=args.actor_targets_root,
        visual_input=visual_input,
        geometry_input=geometry_input,
    )
    try:
        prepared_videos = _bounded_prefetch(
            cache_groups,
            prepare_video,
            workers=args.workers,
            capacity=args.prefetch_videos,
        )
        for video_index, prepared in enumerate(prepared_videos, start=1):
            shard = prepared.shard
            frame_indices = shard["frame_indices"]
            visual_frame_indices = shard["visual_frame_indices"]
            if prepared.visual_images is not None:
                multi_person_videos += 1
                secondary_visual_frames += int(prepared.visual_valid.sum())
                secondary_roi = _secondary_roi_features(
                    adapter,
                    images=prepared.visual_images,
                    boxes=prepared.visual_boxes,
                    valid_mask=prepared.visual_valid,
                    roi_shape=roi_shape,
                    batch_size=args.feature_batch_size,
                    device=device,
                    amp=args.amp,
                )
            else:
                secondary_roi = np.zeros(
                    (len(visual_frame_indices), *roi_shape), dtype=np.float16
                )
            writer.write_video(
                prepared.sample_path,
                frame_indices=frame_indices,
                visual_frame_indices=visual_frame_indices,
                actor_boxes=np.stack(
                    (shard["actor_boxes"], prepared.geometry_boxes), axis=1
                ),
                actor_valid_mask=np.stack(
                    (shard["actor_valid_mask"], prepared.geometry_valid), axis=1
                ),
                visual_actor_boxes=np.stack(
                    (shard["visual_actor_boxes"], prepared.visual_boxes), axis=1
                ),
                global_features=shard["global_features"],
                roi_features=np.stack(
                    (shard["roi_features"], secondary_roi), axis=1
                ),
                source_size=prepared.source_size,
                records=prepared.records,
            )
            if video_index % args.log_interval == 0 or video_index == len(cache_groups):
                elapsed = time.perf_counter() - started
                print(
                    f"videos={video_index}/{len(cache_groups)} "
                    f"multi_person={multi_person_videos} "
                    f"secondary_visual_frames={secondary_visual_frames} "
                    f"videos_per_s={video_index / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        manifest = writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        adapter.close()
    print(f"feature_cache={manifest}", flush=True)
    return manifest


def main() -> None:
    run_expansion(build_parser().parse_args())


if __name__ == "__main__":
    main()
