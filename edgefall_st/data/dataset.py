"""Training dataset that joins OmniFall windows with video supervision."""

from __future__ import annotations

import bisect
import csv
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from edgefall_st.data.labels import FALL_ACTION_ID

DEFAULT_HISTORY_OFFSETS = (0, 1, 2, 4, 8, 16, 24, 31)


@dataclass(frozen=True)
class _VideoSupervision:
    track_frames: tuple[dict[str, Any], ...]
    actor_paths: tuple[tuple[dict[str, Any], ...], ...]
    timestamps: tuple[float, ...]

    @property
    def actor_targets(self) -> tuple[dict[str, Any], ...]:
        return self.actor_paths[0]


@dataclass(frozen=True)
class _PreparedWindow:
    index: int
    window: dict[str, Any]
    sample_path: str
    supervision: _VideoSupervision
    current_position: int
    action_aligned: bool
    history_positions: tuple[int | None, ...]
    target_times: tuple[float, ...]
    video_path: Path


@dataclass(frozen=True)
class _TransformedFrame:
    image_uint8: Tensor
    source_size: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected a JSON object in {path} at line {line_number}"
                    )
                rows.append(row)
    except OSError as error:
        raise OSError(f"Could not read {path}") from error
    return rows


def _read_event_intervals(
    path: Path,
) -> dict[str, tuple[tuple[float, float], ...]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "start_s", "end_s"}
            if reader.fieldnames is None or not required.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Event table is missing required columns {required}: "
                    f"{path}"
                )
            intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row_number, row in enumerate(reader, start=2):
                sample_path = str(row.get("path", "")).replace("\\", "/")
                if not sample_path:
                    raise ValueError(
                        f"Empty event path in {path} at row {row_number}"
                    )
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
                if start_s < 0 or end_s < start_s:
                    raise ValueError(
                        f"Invalid event interval in {path} at row {row_number}"
                    )
                intervals[sample_path].append((start_s, end_s))
            return {
                sample_path: tuple(sorted(values))
                for sample_path, values in intervals.items()
            }
    except OSError as error:
        raise OSError(f"Could not read {path}") from error


def _relative_sample_path(value: object) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Sample path must be relative: {value!r}")
    return path


def _with_suffix(path: Path, suffix: str) -> Path:
    return path if path.suffix else path.with_suffix(suffix)


def _nearest_timestamp_index(
    timestamps: Sequence[float],
    target_time_s: float,
) -> int:
    right = bisect.bisect_left(timestamps, target_time_s)
    if right <= 0:
        return 0
    if right >= len(timestamps):
        return len(timestamps) - 1
    left = right - 1
    if target_time_s - timestamps[left] <= timestamps[right] - target_time_s:
        return left
    return right


def _letterbox_image_uint8(
    image_bgr: np.ndarray,
    input_size: tuple[int, int],
    *,
    fill_value: int,
) -> tuple[Tensor, float, int, int]:
    target_h, target_w = input_size
    source_h, source_w = image_bgr.shape[:2]
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, min(target_w, round(source_w * scale)))
    resized_h = max(1, min(target_h, round(source_h * scale)))
    resized = cv2.resize(
        image_bgr,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    canvas = np.full(
        (target_h, target_w, 3),
        fill_value,
        dtype=np.uint8,
    )
    canvas[
        pad_top : pad_top + resized_h,
        pad_left : pad_left + resized_w,
    ] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return tensor, scale, pad_left, pad_top


def _letterbox_parameters(
    source_size: tuple[int, int],
    input_size: tuple[int, int],
) -> tuple[float, int, int]:
    source_h, source_w = source_size
    target_h, target_w = input_size
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, min(target_w, round(source_w * scale)))
    resized_h = max(1, min(target_h, round(source_h * scale)))
    return (
        scale,
        (target_w - resized_w) // 2,
        (target_h - resized_h) // 2,
    )


def _transform_boxes(
    detections: object,
    *,
    source_size: tuple[int, int],
    input_size: tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
) -> tuple[Tensor, Tensor, Tensor, dict[int, int]]:
    if not isinstance(detections, list):
        raise ValueError("Track frame detections must be a list")

    source_h, source_w = source_size
    target_h, target_w = input_size
    boxes: list[list[float]] = []
    confidences: list[float] = []
    track_ids: list[int] = []
    source_to_output: dict[int, int] = {}
    for detection_index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError("Each detection must be a JSON object")
        values = detection.get("bbox_xyxy_norm")
        output_box = _transform_normalized_box(
            values,
            source_size=source_size,
            input_size=input_size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        if output_box is None:
            continue

        source_to_output[detection_index] = len(boxes)
        boxes.append(output_box)
        confidences.append(float(detection.get("confidence", 0.0)))
        track_id = detection.get("track_id")
        track_ids.append(-1 if track_id is None else int(track_id))

    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
    else:
        box_tensor = torch.empty((0, 4), dtype=torch.float32)
    return (
        box_tensor,
        torch.tensor(confidences, dtype=torch.float32),
        torch.tensor(track_ids, dtype=torch.long),
        source_to_output,
    )


def _transform_normalized_box(
    values: object,
    *,
    source_size: tuple[int, int],
    input_size: tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
) -> list[float] | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    source_box = np.asarray(values, dtype=np.float64)
    if not np.isfinite(source_box).all():
        return None

    source_h, source_w = source_size
    target_h, target_w = input_size
    left = (source_box[0] * source_w * scale + pad_left) / target_w
    top = (source_box[1] * source_h * scale + pad_top) / target_h
    right = (source_box[2] * source_w * scale + pad_left) / target_w
    bottom = (source_box[3] * source_h * scale + pad_top) / target_h
    output_box = np.clip([left, top, right, bottom], 0.0, 1.0)
    if output_box[2] <= output_box[0] or output_box[3] <= output_box[1]:
        return None
    return output_box.tolist()


def _additional_actor_paths(
    track_frames: Sequence[dict[str, Any]],
    primary_targets: Sequence[dict[str, Any]],
    count: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Find label-blind actor paths after removing earlier selections."""

    if count <= 0:
        return ()
    from edgefall_st.pseudo.actor_targets import select_primary_actor

    remaining_frames: list[dict[str, Any]] = []
    for frame, target in zip(track_frames, primary_targets, strict=True):
        detections = frame.get("detections")
        if not isinstance(detections, list):
            raise ValueError("Track frame detections must be a list")
        selected = target.get("actor_detection_index")
        selected_index = None if selected is None else int(selected)
        remaining_frames.append(
            {
                **frame,
                # Candidate discovery must not consume the synthetic action
                # taxonomy because it is unavailable during deployment.
                "action_label": None,
                "detections": [
                    detection
                    for detection_index, detection in enumerate(detections)
                    if detection_index != selected_index
                ],
            }
        )

    paths: list[tuple[dict[str, Any], ...]] = []
    for path_index in range(count):
        if not any(frame["detections"] for frame in remaining_frames):
            empty_path = tuple(
                {"actor_bbox_xyxy_norm": None} for _ in remaining_frames
            )
            paths.extend(empty_path for _ in range(count - path_index))
            break
        targets, _quality = select_primary_actor(remaining_frames)
        paths.append(tuple(targets))
        next_frames: list[dict[str, Any]] = []
        for frame, target in zip(remaining_frames, targets, strict=True):
            selected = target.get("actor_detection_index")
            selected_index = None if selected is None else int(selected)
            detections = frame.get("detections")
            if not isinstance(detections, list):
                raise ValueError("Track frame detections must be a list")
            next_frames.append(
                {
                    **frame,
                    "detections": [
                        detection
                        for detection_index, detection in enumerate(detections)
                        if detection_index != selected_index
                    ],
                }
            )
        remaining_frames = next_frames
    return tuple(paths)


class OmniFallWindowDataset(Dataset[dict[str, Any]]):
    """Load sparse causal video windows and their per-frame supervision.

    ``tracks_root`` and ``actor_targets_root`` point to the directories that
    contain per-video JSONL files, normally the ``tracks`` and ``targets``
    directories produced by the preprocessing pipeline.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        windows_path: str | Path,
        tracks_root: str | Path,
        actor_targets_root: str | Path,
        actor_crops_root: str | Path | None = None,
        actor_crop_size: tuple[int, int] = (128, 128),
        input_size: tuple[int, int] = (288, 512),
        effective_fps: float = 10.0,
        history_offsets: Sequence[int] = DEFAULT_HISTORY_OFFSETS,
        annotation_cache_size: int = 32,
        video_frame_cache_size: int = 4,
        return_uint8: bool = False,
        actor_top_k: int = 1,
        boundary_search_radius: int = 2,
        onset_duration_s: float = 0.5,
        video_suffix: str = ".mp4",
        letterbox_fill: int = 114,
        validate_files: bool = True,
    ) -> None:
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError("input_size dimensions must be positive")
        if actor_crop_size[0] <= 0 or actor_crop_size[1] <= 0:
            raise ValueError("actor_crop_size dimensions must be positive")
        if effective_fps <= 0:
            raise ValueError("effective_fps must be positive")
        offsets = tuple(int(value) for value in history_offsets)
        if not offsets or offsets[0] != 0:
            raise ValueError("history_offsets must start with current offset 0")
        if any(value < 0 for value in offsets):
            raise ValueError("history_offsets cannot contain negative values")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError("history_offsets must be unique and increasing")
        if annotation_cache_size <= 0:
            raise ValueError("annotation_cache_size must be positive")
        if video_frame_cache_size < 0:
            raise ValueError("video_frame_cache_size cannot be negative")
        if boundary_search_radius < 0:
            raise ValueError("boundary_search_radius cannot be negative")
        if onset_duration_s < 0:
            raise ValueError("onset_duration_s cannot be negative")
        if not video_suffix.startswith("."):
            raise ValueError("video_suffix must start with '.'")
        if not 0 <= letterbox_fill <= 255:
            raise ValueError("letterbox_fill must be in [0, 255]")
        if actor_top_k <= 0:
            raise ValueError("actor_top_k must be positive")

        self.dataset_root = Path(dataset_root)
        self.windows_path = Path(windows_path)
        self.tracks_root = Path(tracks_root)
        self.actor_targets_root = Path(actor_targets_root)
        self.actor_crops_root = (
            None if actor_crops_root is None else Path(actor_crops_root)
        )
        self.actor_crop_size = (
            int(actor_crop_size[0]),
            int(actor_crop_size[1]),
        )
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.frame_size = (
            self.actor_crop_size
            if self.actor_crops_root is not None
            else self.input_size
        )
        self.effective_fps = float(effective_fps)
        self.history_offsets = offsets
        self.annotation_cache_size = int(annotation_cache_size)
        self.video_frame_cache_size = int(video_frame_cache_size)
        self.return_uint8 = bool(return_uint8)
        self.actor_top_k = int(actor_top_k)
        self.boundary_search_radius = int(boundary_search_radius)
        self.onset_duration_s = float(onset_duration_s)
        self.video_suffix = video_suffix
        self.letterbox_fill = int(letterbox_fill)
        self.windows = _read_jsonl(self.windows_path)
        if not self.windows:
            raise ValueError(f"No windows found in {self.windows_path}")
        sample_paths = {
            str(window.get("path", ""))
            for window in self.windows
            if str(window.get("path", ""))
        }
        events_path = self.windows_path.with_name("events.csv")
        if events_path.is_file():
            self.video_events = _read_event_intervals(events_path)
            video_has_fall_event = {
                sample_path: sample_path in self.video_events
                for sample_path in sample_paths
            }
        else:
            self.video_events = None
            video_has_fall_event = dict.fromkeys(sample_paths, False)
            for window in self.windows:
                sample_path = str(window.get("path", ""))
                if sample_path:
                    video_has_fall_event[sample_path] |= (
                        int(window.get("action_label", -1)) == FALL_ACTION_ID
                    )
        self.video_has_fall_event = dict(video_has_fall_event)

        self._annotation_cache: OrderedDict[str, _VideoSupervision] = (
            OrderedDict()
        )
        self._video_frame_cache: OrderedDict[
            str,
            dict[int, _TransformedFrame],
        ] = OrderedDict()
        if validate_files:
            self._validate_referenced_files()

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_annotation_cache"] = OrderedDict()
        state["_video_frame_cache"] = OrderedDict()
        return state

    def _paths_for_sample(self, sample_path: object) -> tuple[Path, Path, Path]:
        relative = _relative_sample_path(sample_path)
        video_path = self.dataset_root / _with_suffix(relative, self.video_suffix)
        annotation_path = _with_suffix(relative, ".jsonl")
        return (
            video_path,
            self.tracks_root / annotation_path,
            self.actor_targets_root / annotation_path,
        )

    def _actor_crop_path(self, sample_path: object) -> Path:
        if self.actor_crops_root is None:
            raise RuntimeError("actor_crops_root is not configured")
        relative = _relative_sample_path(sample_path)
        return self.actor_crops_root / _with_suffix(relative, ".npy")

    def _validate_referenced_files(self) -> None:
        required_window_keys = {
            "path",
            "end_frame",
            "end_time_s",
            "action_label",
            "stage_label",
        }
        sample_paths: set[str] = set()
        for index, window in enumerate(self.windows):
            missing = required_window_keys.difference(window)
            if missing:
                raise ValueError(
                    f"Window {index} is missing fields: {sorted(missing)}"
                )
            sample_paths.add(str(window["path"]))

        missing_files: list[Path] = []
        for sample_path in sorted(sample_paths):
            video_path, track_path, actor_path = self._paths_for_sample(
                sample_path
            )
            required_paths = [track_path, actor_path]
            if self.actor_crops_root is None:
                required_paths.insert(0, video_path)
            else:
                required_paths.append(self._actor_crop_path(sample_path))
            for path in required_paths:
                if not path.is_file():
                    missing_files.append(path)
                    if len(missing_files) >= 10:
                        break
            if len(missing_files) >= 10:
                break
        if missing_files:
            formatted = "\n".join(f"- {path}" for path in missing_files)
            raise FileNotFoundError(f"Required dataset files are missing:\n{formatted}")

    def _load_supervision(self, sample_path: str) -> _VideoSupervision:
        cached = self._annotation_cache.get(sample_path)
        if cached is not None:
            self._annotation_cache.move_to_end(sample_path)
            return cached

        _, track_path, actor_path = self._paths_for_sample(sample_path)
        track_frames = _read_jsonl(track_path)
        actor_targets = _read_jsonl(actor_path)
        if not track_frames:
            raise ValueError(f"No track frames found in {track_path}")
        if len(track_frames) != len(actor_targets):
            raise ValueError(
                f"Track/actor row count mismatch for {sample_path}: "
                f"{len(track_frames)} != {len(actor_targets)}"
            )

        timestamps: list[float] = []
        previous_timestamp = float("-inf")
        for position, (frame, actor) in enumerate(
            zip(track_frames, actor_targets, strict=True)
        ):
            frame_index = int(frame.get("frame_index", -1))
            actor_frame_index = int(actor.get("frame_index", -1))
            if frame_index != position:
                raise ValueError(
                    f"Non-contiguous track frame indices for {sample_path}: "
                    f"position {position} has frame_index {frame_index}"
                )
            if actor_frame_index != frame_index:
                raise ValueError(
                    f"Track/actor frame mismatch for {sample_path} at "
                    f"position {position}"
                )
            if str(frame.get("path")) != sample_path:
                raise ValueError(
                    f"Track path mismatch for {sample_path} at position {position}"
                )
            if str(actor.get("path")) != sample_path:
                raise ValueError(
                    f"Actor path mismatch for {sample_path} at position {position}"
                )
            timestamp = float(frame.get("timestamp_s", -1.0))
            actor_timestamp = float(actor.get("timestamp_s", -1.0))
            if timestamp < previous_timestamp:
                raise ValueError(
                    f"Track timestamps are not increasing for {sample_path}"
                )
            if abs(timestamp - actor_timestamp) > 1e-6:
                raise ValueError(
                    f"Track/actor timestamp mismatch for {sample_path} at "
                    f"position {position}"
                )
            timestamps.append(timestamp)
            previous_timestamp = timestamp

        supervision = _VideoSupervision(
            track_frames=tuple(track_frames),
            actor_paths=(
                tuple(actor_targets),
                *_additional_actor_paths(
                    track_frames,
                    actor_targets,
                    self.actor_top_k - 1,
                ),
            ),
            timestamps=tuple(timestamps),
        )
        self._annotation_cache[sample_path] = supervision
        self._annotation_cache.move_to_end(sample_path)
        while len(self._annotation_cache) > self.annotation_cache_size:
            self._annotation_cache.popitem(last=False)
        return supervision

    def _current_frame_position(
        self,
        supervision: _VideoSupervision,
        *,
        end_time_s: float,
        action_label: int,
    ) -> tuple[int, bool]:
        nearest = _nearest_timestamp_index(supervision.timestamps, end_time_s)
        nearest_action = supervision.track_frames[nearest].get("action_label")
        if nearest_action is not None and int(nearest_action) == action_label:
            return nearest, True

        start = max(0, nearest - self.boundary_search_radius)
        stop = min(
            len(supervision.track_frames),
            nearest + self.boundary_search_radius + 1,
        )
        matching = [
            position
            for position in range(start, stop)
            if supervision.track_frames[position].get("action_label") is not None
            and int(
                supervision.track_frames[position]["action_label"]
            )
            == action_label
        ]
        if not matching:
            return nearest, False
        best = min(
            matching,
            key=lambda position: (
                abs(supervision.timestamps[position] - end_time_s),
                position,
            ),
        )
        return best, True

    def _history_positions(
        self,
        supervision: _VideoSupervision,
        *,
        end_time_s: float,
        current_position: int,
    ) -> tuple[list[int | None], list[float]]:
        positions: list[int | None] = []
        target_times: list[float] = []
        for offset in self.history_offsets:
            target_time = end_time_s - offset / self.effective_fps
            target_times.append(target_time)
            if target_time < 0:
                positions.append(None)
            elif offset == 0:
                positions.append(current_position)
            else:
                positions.append(
                    _nearest_timestamp_index(
                        supervision.timestamps,
                        target_time,
                    )
                )
        return positions, target_times

    def _window_onset_target(
        self,
        supervision: _VideoSupervision,
        *,
        current_position: int,
        end_time_s: float,
        action_label: int,
        action_aligned: bool,
    ) -> float:
        if action_label != FALL_ACTION_ID or not action_aligned:
            return 0.0
        segment_start = current_position
        while segment_start > 0:
            previous_action = supervision.track_frames[
                segment_start - 1
            ].get("action_label")
            if previous_action is None or int(previous_action) != FALL_ACTION_ID:
                break
            segment_start -= 1
        onset_time_s = supervision.timestamps[segment_start]
        elapsed = end_time_s - onset_time_s
        return float(-1e-6 <= elapsed <= self.onset_duration_s + 1e-6)

    def _decode_frames(
        self,
        video_path: Path,
        positions: Sequence[int | None],
    ) -> dict[int, np.ndarray]:
        requested = sorted({value for value in positions if value is not None})
        if not requested:
            return {}
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        decoded: dict[int, np.ndarray] = {}
        requested_set = set(requested)
        first_frame = requested[0]
        last_frame = requested[-1]
        try:
            span = last_frame - first_frame + 1
            use_sequential_decode = span <= len(requested) * 4
            if use_sequential_decode:
                capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
                for frame_index in range(first_frame, last_frame + 1):
                    success, image = capture.read()
                    if not success or image is None:
                        raise RuntimeError(
                            f"Could not decode frame {frame_index} "
                            f"from {video_path}"
                        )
                    if frame_index in requested_set:
                        decoded[frame_index] = image
            else:
                for frame_index in requested:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, image = capture.read()
                    if not success or image is None:
                        raise RuntimeError(
                            f"Could not decode frame {frame_index} "
                            f"from {video_path}"
                        )
                    decoded[frame_index] = image
        finally:
            capture.release()
        return decoded

    def _prepare_window(self, index: int) -> _PreparedWindow:
        window = self.windows[index]
        sample_path = str(window["path"])
        end_time_s = float(window["end_time_s"])
        action_label = int(window["action_label"])
        stage_label = int(window["stage_label"])
        supervision = self._load_supervision(sample_path)
        current_position, action_aligned = self._current_frame_position(
            supervision,
            end_time_s=end_time_s,
            action_label=action_label,
        )
        history_positions, target_times = self._history_positions(
            supervision,
            end_time_s=end_time_s,
            current_position=current_position,
        )
        video_path, _, _ = self._paths_for_sample(sample_path)
        return _PreparedWindow(
            index=int(index),
            window=window,
            sample_path=sample_path,
            supervision=supervision,
            current_position=current_position,
            action_aligned=action_aligned,
            history_positions=tuple(history_positions),
            target_times=tuple(target_times),
            video_path=video_path,
        )

    def _transformed_frames(
        self,
        prepared_windows: Sequence[_PreparedWindow],
    ) -> dict[int, _TransformedFrame]:
        if not prepared_windows:
            raise ValueError("prepared_windows cannot be empty")
        sample_path = prepared_windows[0].sample_path
        if any(
            prepared.sample_path != sample_path
            for prepared in prepared_windows
        ):
            raise ValueError("Frame batches must contain one video path")

        cached = self._video_frame_cache.get(sample_path)
        if cached is None:
            cached = {}
            if self.video_frame_cache_size > 0:
                self._video_frame_cache[sample_path] = cached
        elif self.video_frame_cache_size > 0:
            self._video_frame_cache.move_to_end(sample_path)

        requested = {
            position
            for prepared in prepared_windows
            for position in prepared.history_positions
            if position is not None
        }
        missing = sorted(requested.difference(cached))
        if missing:
            supervision = prepared_windows[0].supervision
            if self.actor_crops_root is not None:
                crop_path = self._actor_crop_path(sample_path)
                crops = np.load(crop_path, allow_pickle=False)
                expected_shape = (
                    len(supervision.track_frames),
                    3,
                    *self.actor_crop_size,
                )
                if crops.dtype != np.uint8 or crops.shape != expected_shape:
                    raise ValueError(
                        f"Invalid actor crop cache {crop_path}: "
                        f"expected uint8 {expected_shape}, got "
                        f"{crops.dtype} {crops.shape}"
                    )
                frame_indices = range(len(supervision.track_frames))
                decoded: dict[int, np.ndarray] | None = None
            else:
                decoded = self._decode_frames(
                    prepared_windows[0].video_path,
                    missing,
                )
                frame_indices = missing

            for frame_index in frame_indices:
                track_frame = supervision.track_frames[frame_index]
                source_h = int(track_frame["height"])
                source_w = int(track_frame["width"])
                if decoded is None:
                    image_uint8 = torch.from_numpy(
                        np.array(crops[frame_index], copy=True)
                    )
                    scale, pad_left, pad_top = _letterbox_parameters(
                        (source_h, source_w),
                        self.input_size,
                    )
                else:
                    image = decoded[frame_index]
                    if image.shape[:2] != (source_h, source_w):
                        raise ValueError(
                            f"Video/track dimensions disagree for "
                            f"{sample_path} at frame {frame_index}: "
                            f"{image.shape[1]}x{image.shape[0]} != "
                            f"{source_w}x{source_h}"
                        )
                    image_uint8, scale, pad_left, pad_top = (
                        _letterbox_image_uint8(
                            image,
                            self.input_size,
                            fill_value=self.letterbox_fill,
                        )
                    )
                cached[frame_index] = _TransformedFrame(
                    image_uint8=image_uint8,
                    source_size=(source_h, source_w),
                    scale=scale,
                    pad_left=pad_left,
                    pad_top=pad_top,
                )

        if self.video_frame_cache_size > 0:
            self._video_frame_cache.move_to_end(sample_path)
            while (
                len(self._video_frame_cache) > self.video_frame_cache_size
            ):
                self._video_frame_cache.popitem(last=False)
        return cached

    def _build_sample(
        self,
        prepared: _PreparedWindow,
        transformed_frames: dict[int, _TransformedFrame],
    ) -> dict[str, Any]:
        window = prepared.window
        sample_path = prepared.sample_path
        supervision = prepared.supervision
        current_position = prepared.current_position
        history_positions = prepared.history_positions
        action_aligned = prepared.action_aligned
        end_time_s = float(window["end_time_s"])
        action_label = int(window["action_label"])
        stage_label = int(window["stage_label"])
        if self.video_events is None:
            event_target = stage_label != 0
        else:
            event_target = any(
                start_s <= end_time_s < end_s
                for start_s, end_s in self.video_events.get(sample_path, ())
            )

        current_frame = supervision.track_frames[current_position]
        source_h = int(current_frame["height"])
        source_w = int(current_frame["width"])
        if source_h <= 0 or source_w <= 0:
            raise ValueError(f"Invalid source dimensions for {sample_path}")

        image_tensors: list[Tensor] = []
        valid_values: list[float] = []
        actor_boxes: list[Tensor] = []
        actor_valid_values: list[list[float]] = []
        image_dtype = torch.uint8 if self.return_uint8 else torch.float32
        current_transform: tuple[float, int, int] | None = None
        for slot, position in enumerate(history_positions):
            if position is None:
                image_tensors.append(
                    torch.zeros((3, *self.frame_size), dtype=image_dtype)
                )
                valid_values.append(0.0)
                actor_boxes.append(
                    torch.zeros(
                        (self.actor_top_k, 4), dtype=torch.float32
                    )
                )
                actor_valid_values.append([0.0] * self.actor_top_k)
                continue
            transformed = transformed_frames[position]
            if transformed.source_size != (source_h, source_w):
                raise ValueError(
                    f"Video/track dimensions disagree for {sample_path} at "
                    f"frame {position}"
                )
            if self.return_uint8:
                image_tensors.append(transformed.image_uint8)
            else:
                image_tensors.append(
                    transformed.image_uint8.to(torch.float32).div_(255.0)
                )
            valid_values.append(1.0)
            if slot == 0:
                current_transform = (
                    transformed.scale,
                    transformed.pad_left,
                    transformed.pad_top,
                )
            slot_boxes: list[Tensor] = []
            slot_valid: list[float] = []
            for actor_path_index, actor_path in enumerate(
                supervision.actor_paths
            ):
                actor_target = actor_path[position]
                actor_values = actor_target.get("actor_bbox_xyxy_norm")
                if actor_values is None and actor_path_index == 0:
                    detection_index = actor_target.get(
                        "actor_detection_index"
                    )
                    detections = supervision.track_frames[position].get(
                        "detections"
                    )
                    if (
                        detection_index is not None
                        and isinstance(detections, list)
                        and 0 <= int(detection_index) < len(detections)
                        and isinstance(
                            detections[int(detection_index)], dict
                        )
                    ):
                        actor_values = detections[
                            int(detection_index)
                        ].get("bbox_xyxy_norm")
                transformed_actor_box = _transform_normalized_box(
                    actor_values,
                    source_size=transformed.source_size,
                    input_size=self.input_size,
                    scale=transformed.scale,
                    pad_left=transformed.pad_left,
                    pad_top=transformed.pad_top,
                )
                if transformed_actor_box is None:
                    slot_boxes.append(torch.zeros(4, dtype=torch.float32))
                    slot_valid.append(0.0)
                else:
                    slot_boxes.append(
                        torch.tensor(
                            transformed_actor_box, dtype=torch.float32
                        )
                    )
                    slot_valid.append(1.0)
            actor_boxes.append(torch.stack(slot_boxes))
            actor_valid_values.append(slot_valid)

        if current_transform is None:
            raise RuntimeError("The current frame cannot be a padded history slot")
        scale, pad_left, pad_top = current_transform
        boxes, confidences, track_ids, source_to_output = _transform_boxes(
            current_frame.get("detections"),
            source_size=(source_h, source_w),
            input_size=self.input_size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )

        actor = supervision.actor_targets[current_position]
        source_actor_index = actor.get("actor_detection_index")
        actor_box_index = (
            -1
            if source_actor_index is None
            else source_to_output.get(int(source_actor_index), -1)
        )
        action_loss_weight = float(actor.get("action_loss_weight", 0.0))
        track_loss_weight = float(actor.get("track_loss_weight", 0.0))
        if actor_box_index < 0 or not action_aligned:
            action_loss_weight = 0.0
            track_loss_weight = 0.0

        actor_track_id = actor.get("actor_source_track_id")
        actor_identity = actor.get("actor_identity")
        onset_target = self._window_onset_target(
            supervision,
            current_position=current_position,
            end_time_s=end_time_s,
            action_label=action_label,
            action_aligned=action_aligned,
        )
        target = {
            "boxes": boxes,
            "box_confidences": confidences,
            "source_track_ids": track_ids,
            "actor_box_index": torch.tensor(actor_box_index, dtype=torch.long),
            "actor_source_track_id": torch.tensor(
                -1 if actor_track_id is None else int(actor_track_id),
                dtype=torch.long,
            ),
            "actor_identity": torch.tensor(
                -1 if actor_identity is None else int(actor_identity),
                dtype=torch.long,
            ),
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "stage_label": torch.tensor(stage_label, dtype=torch.long),
            "event_target": torch.tensor(
                event_target,
                dtype=torch.float32,
            ),
            "onset_target": torch.tensor(
                onset_target,
                dtype=torch.float32,
            ),
            "source_onset_target": torch.tensor(
                float(actor.get("onset_target", 0.0)),
                dtype=torch.float32,
            ),
            "action_loss_weight": torch.tensor(
                action_loss_weight,
                dtype=torch.float32,
            ),
            "track_loss_weight": torch.tensor(
                track_loss_weight,
                dtype=torch.float32,
            ),
        }
        metadata = {
            "path": sample_path,
            "video_has_fall_event": self.video_has_fall_event[sample_path],
            "event_target": event_target,
            "window_index": int(window.get("window_index", prepared.index)),
            "end_frame": int(window["end_frame"]),
            "end_time_s": end_time_s,
            "source_frame_index": int(current_frame["frame_index"]),
            "source_timestamp_s": float(current_frame["timestamp_s"]),
            "history_frame_indices": [
                -1 if value is None else int(value)
                for value in history_positions
            ],
            "history_target_times_s": list(prepared.target_times),
            "source_size": (source_h, source_w),
            "input_size": self.input_size,
            "frame_size": self.frame_size,
            "pre_cropped_input": self.actor_crops_root is not None,
            "window_action_label": action_label,
            "source_action_label": current_frame.get("action_label"),
            "action_aligned": action_aligned,
        }
        return {
            "images": torch.stack(image_tensors),
            "valid_mask": torch.tensor(valid_values, dtype=torch.float32),
            "actor_boxes": (
                torch.stack(actor_boxes)[:, 0]
                if self.actor_top_k == 1
                else torch.stack(actor_boxes)
            ),
            "actor_valid_mask": (
                torch.tensor(actor_valid_values, dtype=torch.float32)[:, 0]
                if self.actor_top_k == 1
                else torch.tensor(actor_valid_values, dtype=torch.float32)
            ),
            "target": target,
            "metadata": metadata,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        prepared = self._prepare_window(index)
        transformed = self._transformed_frames([prepared])
        return self._build_sample(prepared, transformed)

    def __getitems__(
        self,
        indices: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Bulk-load a DataLoader batch with one decode pass per video."""

        prepared = [self._prepare_window(int(index)) for index in indices]
        grouped: dict[str, list[tuple[int, _PreparedWindow]]] = defaultdict(list)
        for output_index, item in enumerate(prepared):
            grouped[item.sample_path].append((output_index, item))

        results: list[dict[str, Any] | None] = [None] * len(prepared)
        for items in grouped.values():
            group_windows = [item for _, item in items]
            transformed = self._transformed_frames(group_windows)
            for output_index, item in items:
                results[output_index] = self._build_sample(item, transformed)
        if any(result is None for result in results):
            raise RuntimeError("Bulk dataset loading produced a missing sample")
        return [result for result in results if result is not None]


def omnifall_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size model inputs while preserving variable box targets."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    return {
        "images": torch.stack([sample["images"] for sample in batch]),
        "valid_mask": torch.stack(
            [sample["valid_mask"] for sample in batch]
        ),
        "actor_boxes": torch.stack(
            [sample["actor_boxes"] for sample in batch]
        ),
        "actor_valid_mask": torch.stack(
            [sample["actor_valid_mask"] for sample in batch]
        ),
        "targets": [sample["target"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }


def omnifall_indexed_collate_fn(
    batch: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Collate windows into one unique-frame pool and history index table."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(batch)
    slots = int(batch[0]["images"].shape[0])
    multi_person = batch[0]["actor_boxes"].ndim == 3
    actor_shape = tuple(batch[0]["actor_boxes"].shape[1:])
    actor_mask_shape = tuple(batch[0]["actor_valid_mask"].shape[1:])
    history_indices = torch.full(
        (batch_size, slots),
        -1,
        dtype=torch.long,
    )
    unique_images: list[Tensor] = []
    unique_actor_boxes: list[Tensor] = []
    unique_actor_valid_values: list[Tensor] = []
    unique_index_by_frame: dict[tuple[str, int], int] = {}

    for sample_index, sample in enumerate(batch):
        images = sample["images"]
        valid_mask = sample["valid_mask"]
        actor_boxes = sample["actor_boxes"]
        actor_valid_mask = sample["actor_valid_mask"]
        frame_indices = sample["metadata"]["history_frame_indices"]
        if images.ndim != 4 or images.shape[0] != slots:
            raise ValueError("All samples must contain K images with shape [K,C,H,W]")
        expected_boxes = (slots, *actor_shape)
        expected_actor_mask = (slots, *actor_mask_shape)
        if (
            valid_mask.shape != (slots,)
            or actor_boxes.shape != expected_boxes
            or actor_valid_mask.shape != expected_actor_mask
            or len(frame_indices) != slots
            or (multi_person and actor_shape[-1] != 4)
            or (not multi_person and actor_shape != (4,))
        ):
            raise ValueError(
                "History images, boxes, masks, and metadata disagree"
            )

        sample_path = str(sample["metadata"]["path"])
        for slot_index, frame_index_value in enumerate(frame_indices):
            frame_index = int(frame_index_value)
            is_valid = bool(valid_mask[slot_index].item())
            if not is_valid:
                if frame_index >= 0:
                    raise ValueError("Invalid history slots must use frame index -1")
                continue
            if frame_index < 0:
                raise ValueError("Valid history slots require a source frame index")

            key = (sample_path, frame_index)
            unique_index = unique_index_by_frame.get(key)
            if unique_index is None:
                unique_index = len(unique_images)
                unique_index_by_frame[key] = unique_index
                unique_images.append(images[slot_index])
                unique_actor_boxes.append(actor_boxes[slot_index])
                unique_actor_valid_values.append(
                    actor_valid_mask[slot_index].clone()
                )
            else:
                existing_valid = unique_actor_valid_values[unique_index]
                current_valid = actor_valid_mask[slot_index]
                if not torch.equal(existing_valid, current_valid) or not torch.allclose(
                    unique_actor_boxes[unique_index],
                    actor_boxes[slot_index],
                ):
                    raise ValueError(
                        "Shared source frames have inconsistent actor boxes"
                    )
            history_indices[sample_index, slot_index] = unique_index

    if not unique_images:
        raise ValueError("A batch must contain at least one valid frame")

    return {
        "images": torch.stack(unique_images),
        "actor_boxes": torch.stack(unique_actor_boxes),
        "actor_valid_mask": torch.stack(unique_actor_valid_values).float(),
        "history_indices": history_indices,
        "valid_mask": torch.stack(
            [sample["valid_mask"] for sample in batch]
        ),
        "targets": [sample["target"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }
