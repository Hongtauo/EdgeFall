"""Per-video frozen detector feature cache for V3.4 training."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


CACHE_FORMAT = "edgefall-frozen-features-v2"


def roi_feature_shape(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return the per-person ROI shape for single- or multi-person caches."""

    shape = tuple(int(value) for value in manifest["arrays"]["roi_features"])
    if len(shape) == 3:
        return shape
    if len(shape) == 4:
        return shape[1:]
    raise ValueError("ROI feature cache shape must be [C,H,W] or [P,C,H,W]")


def cache_person_capacity(manifest: Mapping[str, Any]) -> int:
    shape = tuple(int(value) for value in manifest["arrays"]["actor_boxes"])
    if shape == (4,):
        return 1
    if len(shape) == 2 and shape[1] == 4:
        return shape[0]
    raise ValueError("Actor box cache shape must be [4] or [P,4]")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _relative_video_cache_path(sample_path: str) -> Path:
    relative = Path(sample_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Video path must be relative: {sample_path!r}")
    return relative.with_suffix(".npz")


class FrozenFeatureCacheWriter:
    """Write each source frame once, then index overlapping windows by frame."""

    def __init__(
        self,
        root: str | Path,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.videos_root = self.root / "videos"
        self.videos_root.mkdir()
        self.index_path = self.root / "index.jsonl"
        self._index_handle = self.index_path.open("w", encoding="utf-8")
        self.metadata = dict(metadata)
        self.sample_count = 0
        self.video_count = 0
        self.geometry_frame_count = 0
        self.visual_frame_count = 0
        self._seen_paths: set[str] = set()
        self._array_shapes: dict[str, list[int]] | None = None

    def write_video(
        self,
        sample_path: str,
        *,
        frame_indices: np.ndarray,
        visual_frame_indices: np.ndarray,
        actor_boxes: np.ndarray,
        actor_valid_mask: np.ndarray,
        visual_actor_boxes: np.ndarray,
        global_features: np.ndarray,
        roi_features: np.ndarray,
        source_size: tuple[int, int],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        if sample_path in self._seen_paths:
            raise ValueError(f"Video was written twice: {sample_path!r}")
        if not records:
            raise ValueError("A cached video must contain at least one window")
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        visual_frame_indices = np.asarray(visual_frame_indices, dtype=np.int64)
        actor_boxes = np.asarray(actor_boxes, dtype=np.float32)
        actor_valid_mask = np.asarray(actor_valid_mask, dtype=np.float32)
        visual_actor_boxes = np.asarray(visual_actor_boxes, dtype=np.float32)
        global_features = np.asarray(global_features, dtype=np.float16)
        roi_features = np.asarray(roi_features, dtype=np.float16)
        if frame_indices.ndim != 1 or visual_frame_indices.ndim != 1:
            raise ValueError("Frame indices must be one-dimensional")
        if frame_indices.size == 0 or visual_frame_indices.size == 0:
            raise ValueError("Frame index arrays cannot be empty")
        if not np.all(frame_indices[:-1] < frame_indices[1:]):
            raise ValueError("frame_indices must be unique and increasing")
        if not np.all(
            visual_frame_indices[:-1] < visual_frame_indices[1:]
        ):
            raise ValueError(
                "visual_frame_indices must be unique and increasing"
            )
        if not set(visual_frame_indices.tolist()).issubset(
            frame_indices.tolist()
        ):
            raise ValueError("Visual frames must be a subset of geometry frames")
        if actor_boxes.shape[0] != frame_indices.size or actor_boxes.shape[-1] != 4:
            raise ValueError("actor_boxes and frame_indices disagree")
        if actor_valid_mask.shape != actor_boxes.shape[:-1]:
            raise ValueError("actor_valid_mask and frame_indices disagree")
        if (
            visual_actor_boxes.shape[0] != visual_frame_indices.size
            or visual_actor_boxes.shape[-1] != 4
        ):
            raise ValueError("visual_actor_boxes and visual frames disagree")
        if global_features.shape[0] != visual_frame_indices.size:
            raise ValueError("global_features and visual frames disagree")
        if roi_features.shape[0] != visual_frame_indices.size:
            raise ValueError("roi_features and visual frames disagree")
        if actor_boxes.ndim not in (2, 3):
            raise ValueError("actor_boxes must be [T,4] or [T,P,4]")
        if visual_actor_boxes.ndim != actor_boxes.ndim:
            raise ValueError("Geometry and visual actor ranks disagree")
        if actor_boxes.ndim == 3:
            if actor_boxes.shape[1] != visual_actor_boxes.shape[1]:
                raise ValueError("Geometry and visual person counts disagree")
            if roi_features.ndim != 5 or roi_features.shape[1] != actor_boxes.shape[1]:
                raise ValueError("ROI features and actor person counts disagree")
        elif roi_features.ndim != 4:
            raise ValueError("Single-person ROI features must be [T,C,H,W]")

        shapes = {
            "global_features": list(global_features.shape[1:]),
            "roi_features": list(roi_features.shape[1:]),
            "actor_boxes": list(actor_boxes.shape[1:]),
            "actor_valid_mask": list(actor_valid_mask.shape[1:]),
            "visual_actor_boxes": list(visual_actor_boxes.shape[1:]),
        }
        if self._array_shapes is None:
            self._array_shapes = shapes
        elif self._array_shapes != shapes:
            raise ValueError("Feature array shapes changed between videos")

        relative_cache = _relative_video_cache_path(sample_path)
        cache_path = self.videos_root / relative_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            np.savez(
                handle,
                frame_indices=frame_indices,
                visual_frame_indices=visual_frame_indices,
                actor_boxes=actor_boxes,
                actor_valid_mask=actor_valid_mask,
                visual_actor_boxes=visual_actor_boxes,
                global_features=global_features,
                roi_features=roi_features,
                source_size=np.asarray(source_size, dtype=np.int32),
            )
        for record in records:
            if str(record.get("path")) != sample_path:
                raise ValueError("Window record path does not match video path")
            payload = {
                **_json_value(record),
                "video_cache": relative_cache.as_posix(),
            }
            self._index_handle.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
        self._index_handle.flush()
        self._seen_paths.add(sample_path)
        self.sample_count += len(records)
        self.video_count += 1
        self.geometry_frame_count += int(frame_indices.size)
        self.visual_frame_count += int(visual_frame_indices.size)

    def close(self) -> Path:
        if self._index_handle.closed:
            return self.root / "manifest.json"
        self._index_handle.close()
        if self.sample_count <= 0 or self._array_shapes is None:
            raise RuntimeError("Cannot finalize an empty feature cache")
        manifest = {
            "format": CACHE_FORMAT,
            "complete": True,
            "sample_count": self.sample_count,
            "video_count": self.video_count,
            "geometry_frame_count": self.geometry_frame_count,
            "visual_frame_count": self.visual_frame_count,
            "arrays": self._array_shapes,
            **_json_value(self.metadata),
        }
        path = self.root / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return path

    def abort(self) -> None:
        if not self._index_handle.closed:
            self._index_handle.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}"
                )
            rows.append(value)
    return rows


class FrozenFeatureWindowDataset(Dataset[dict[str, Any]]):
    """Read V3.4 windows while loading each per-video feature file once."""

    def __init__(
        self,
        root: str | Path,
        *,
        shard_cache_size: int = 2,
        selected_indices: Sequence[int] | None = None,
        selected_paths: Collection[str] | None = None,
        domain: str | None = None,
    ) -> None:
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        self.root = Path(root)
        with (self.root / "manifest.json").open(
            "r", encoding="utf-8"
        ) as handle:
            manifest = json.load(handle)
        if manifest.get("format") != CACHE_FORMAT or not manifest.get(
            "complete", False
        ):
            raise ValueError(f"Incomplete or unsupported cache: {self.root}")
        entries = _read_jsonl(self.root / "index.jsonl")
        if len(entries) != int(manifest["sample_count"]):
            raise ValueError("Feature cache manifest/index count mismatch")
        if selected_indices is not None:
            entries = [entries[int(index)] for index in selected_indices]
        if selected_paths is not None:
            allowed = {str(value) for value in selected_paths}
            entries = [entry for entry in entries if str(entry["path"]) in allowed]
        if not entries:
            raise ValueError("Feature cache selection is empty")
        self.manifest = manifest
        self.domain = str(domain or manifest.get("domain", "omnifall"))
        self.entries = entries
        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.windows = [
            {
                "path": entry["path"],
                "window_index": entry["window_index"],
                "end_time_s": entry["end_time_s"],
                "action_label": entry["action_label"],
                "stage_label": entry["stage_label"],
                "event_target": entry["event_target"],
                "weak_video_label": bool(entry.get("weak_video_label", False)),
                "domain": str(entry.get("domain", self.domain)),
            }
            for entry in entries
        ]
        self.video_has_fall_event: dict[str, bool] = {}
        for entry in entries:
            path = str(entry["path"])
            value = bool(entry["video_has_fall_event"])
            previous = self.video_has_fall_event.get(path)
            if previous is not None and previous != value:
                raise ValueError(f"Inconsistent video target for {path!r}")
            self.video_has_fall_event[path] = value
        self._visual_slots = tuple(
            self.history_offsets.index(value) for value in self.visual_offsets
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state

    @property
    def history_offsets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.manifest["history_offsets"])

    @property
    def visual_offsets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.manifest["visual_offsets"])

    def _load_shard(self, name: str) -> dict[str, Any]:
        cached = self._shard_cache.get(name)
        if cached is not None:
            self._shard_cache.move_to_end(name)
            return cached
        path = self.root / "videos" / name
        with np.load(path, allow_pickle=False) as archive:
            cached = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
            }
        cached["row_by_frame"] = {
            int(frame): index
            for index, frame in enumerate(cached["frame_indices"])
        }
        cached["visual_row_by_frame"] = {
            int(frame): index
            for index, frame in enumerate(cached["visual_frame_indices"])
        }
        self._shard_cache[name] = cached
        self._shard_cache.move_to_end(name)
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        shard = self._load_shard(str(entry["video_cache"]))
        history_frames = [int(value) for value in entry["history_frame_indices"]]
        if len(history_frames) != len(self.history_offsets):
            raise ValueError("Cached history slot count is inconsistent")
        actor_boxes: list[np.ndarray] = []
        actor_mask: list[np.ndarray | float] = []
        actor_shape = tuple(self.manifest["arrays"]["actor_boxes"])
        actor_mask_shape = tuple(
            self.manifest["arrays"]["actor_valid_mask"]
        )
        for frame in history_frames:
            if frame < 0:
                actor_boxes.append(np.zeros(actor_shape, dtype=np.float32))
                if actor_mask_shape:
                    actor_mask.append(
                        np.zeros(actor_mask_shape, dtype=np.float32)
                    )
                else:
                    actor_mask.append(0.0)
                continue
            row = shard["row_by_frame"].get(frame)
            if row is None:
                raise ValueError(f"Geometry frame {frame} is missing from cache")
            actor_boxes.append(shard["actor_boxes"][row])
            if actor_mask_shape:
                actor_mask.append(shard["actor_valid_mask"][row])
            else:
                actor_mask.append(float(shard["actor_valid_mask"][row]))

        global_features: list[np.ndarray] = []
        roi_features: list[np.ndarray] = []
        global_shape = tuple(self.manifest["arrays"]["global_features"])
        roi_shape = tuple(self.manifest["arrays"]["roi_features"])
        for slot in self._visual_slots:
            frame = history_frames[slot]
            if frame < 0:
                global_features.append(np.zeros(global_shape, dtype=np.float16))
                roi_features.append(np.zeros(roi_shape, dtype=np.float16))
                continue
            row = shard["visual_row_by_frame"].get(frame)
            if row is None:
                raise ValueError(f"Visual frame {frame} is missing from cache")
            global_features.append(shard["global_features"][row])
            roi_features.append(shard["roi_features"][row])

        target = {
            "action_label": torch.tensor(int(entry["action_label"]), dtype=torch.long),
            "stage_label": torch.tensor(int(entry["stage_label"]), dtype=torch.long),
            "event_target": torch.tensor(float(entry["event_target"]), dtype=torch.float32),
            "onset_target": torch.tensor(float(entry["onset_target"]), dtype=torch.float32),
            "action_loss_weight": torch.tensor(
                float(entry["action_loss_weight"]), dtype=torch.float32
            ),
            "event_loss_weight": torch.tensor(
                float(
                    entry.get(
                        "event_loss_weight", entry["action_loss_weight"]
                    )
                ),
                dtype=torch.float32,
            ),
            "control_loss_weight": torch.tensor(
                float(
                    entry.get(
                        "control_loss_weight", entry["action_loss_weight"]
                    )
                ),
                dtype=torch.float32,
            ),
        }
        metadata = {
            "path": entry["path"],
            "video_has_fall_event": entry["video_has_fall_event"],
            "event_target": entry["event_target"],
            "window_index": entry["window_index"],
            "end_frame": entry["end_frame"],
            "end_time_s": entry["end_time_s"],
            "window_action_label": entry["window_action_label"],
            "weak_video_label": bool(entry.get("weak_video_label", False)),
            "domain": str(entry.get("domain", self.domain)),
        }
        valid_mask = np.asarray(
            [float(frame >= 0) for frame in history_frames], dtype=np.float32
        )
        return {
            "global_features": torch.from_numpy(np.stack(global_features)),
            "roi_features": torch.from_numpy(np.stack(roi_features)),
            "actor_boxes": torch.from_numpy(np.stack(actor_boxes)).float(),
            "valid_mask": torch.from_numpy(valid_mask),
            "actor_valid_mask": torch.from_numpy(
                np.asarray(actor_mask, dtype=np.float32)
            ),
            "target": target,
            "metadata": metadata,
        }


class MultiDomainFrozenFeatureWindowDataset(Dataset[dict[str, Any]]):
    """Present multiple contract-compatible feature caches as one dataset."""

    def __init__(self, datasets: Sequence[FrozenFeatureWindowDataset]) -> None:
        if not datasets:
            raise ValueError("At least one feature cache is required")
        self.datasets = tuple(datasets)
        self.manifest = self.datasets[0].manifest
        self.manifests = tuple(dataset.manifest for dataset in self.datasets)
        self.windows = [
            window for dataset in self.datasets for window in dataset.windows
        ]
        self.video_has_fall_event: dict[str, bool] = {}
        self.weak_positive_paths: set[str] = set()
        seen_paths: set[str] = set()
        for dataset in self.datasets:
            overlap = seen_paths.intersection(dataset.video_has_fall_event)
            if overlap:
                rendered = ", ".join(sorted(overlap)[:3])
                raise ValueError(f"Video paths collide across caches: {rendered}")
            seen_paths.update(dataset.video_has_fall_event)
            self.video_has_fall_event.update(dataset.video_has_fall_event)
            self.weak_positive_paths.update(
                str(window["path"])
                for window in dataset.windows
                if bool(window.get("weak_video_label", False))
                and dataset.video_has_fall_event[str(window["path"])]
            )
        total = 0
        self._ends: list[int] = []
        for dataset in self.datasets:
            total += len(dataset)
            self._ends.append(total)

    @property
    def history_offsets(self) -> tuple[int, ...]:
        return self.datasets[0].history_offsets

    @property
    def visual_offsets(self) -> tuple[int, ...]:
        return self.datasets[0].visual_offsets

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self._ends, index)
        start = 0 if dataset_index == 0 else self._ends[dataset_index - 1]
        return self.datasets[dataset_index][index - start]


def frozen_feature_collate_fn(
    batch: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    multi_person = any(sample["actor_boxes"].ndim == 3 for sample in batch)
    if not multi_person:
        tensors = {
            name: torch.stack([sample[name] for sample in batch])
            for name in (
                "global_features",
                "roi_features",
                "actor_boxes",
                "valid_mask",
                "actor_valid_mask",
            )
        }
    else:
        person_count = max(
            1 if sample["actor_boxes"].ndim == 2 else sample["actor_boxes"].shape[1]
            for sample in batch
        )

        def pad_people(value: Tensor, *, person_axis: int) -> Tensor:
            current = int(value.shape[person_axis])
            if current == person_count:
                return value
            shape = list(value.shape)
            shape[person_axis] = person_count - current
            return torch.cat((value, value.new_zeros(shape)), dim=person_axis)

        roi_values: list[Tensor] = []
        actor_values: list[Tensor] = []
        actor_masks: list[Tensor] = []
        for sample in batch:
            roi = sample["roi_features"]
            boxes = sample["actor_boxes"]
            mask = sample["actor_valid_mask"]
            if boxes.ndim == 2:
                roi = roi.unsqueeze(1)
                boxes = boxes.unsqueeze(1)
                mask = mask.unsqueeze(1)
            roi_values.append(pad_people(roi, person_axis=1))
            actor_values.append(pad_people(boxes, person_axis=1))
            actor_masks.append(pad_people(mask, person_axis=1))
        tensors = {
            "global_features": torch.stack(
                [sample["global_features"] for sample in batch]
            ),
            "roi_features": torch.stack(roi_values),
            "actor_boxes": torch.stack(actor_values),
            "valid_mask": torch.stack(
                [sample["valid_mask"] for sample in batch]
            ),
            "actor_valid_mask": torch.stack(actor_masks),
        }
    return tensors | {
        "targets": [sample["target"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }
