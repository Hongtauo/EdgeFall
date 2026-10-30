"""Batched multi-video YOLO inference with one independent tracker per video."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from edgefall_st.data.annotations import VideoAnnotations, label_at
from edgefall_st.pseudo.quality import VideoTrackQuality
from edgefall_st.pseudo.yolo26_tracker import TeacherConfig, _draw_frame


@dataclass(frozen=True)
class BatchVideoTask:
    video_path: Path
    output_jsonl: Path
    relative_path: str
    annotations: VideoAnnotations | None = None
    visualization_path: Path | None = None


TrackerFactory = Callable[[TeacherConfig], object]


def _build_bytetracker(config: TeacherConfig):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    tracker_path = check_yaml(config.tracker)
    arguments = IterableSimpleNamespace(**YAML.load(tracker_path))
    if arguments.tracker_type != "bytetrack":
        raise ValueError(
            "Batched pseudo labeling currently supports only ByteTrack, "
            f"got {arguments.tracker_type!r}"
        )
    arguments.device = config.device
    return BYTETracker(args=arguments)


def _temporary_video_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


class _VideoState:
    def __init__(self, task: BatchVideoTask, tracker: object) -> None:
        self.task = task
        self.tracker = tracker
        self.capture = cv2.VideoCapture(str(task.video_path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {task.video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.expected_frames = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            self.fps <= 0
            or self.expected_frames <= 0
            or self.width <= 0
            or self.height <= 0
        ):
            self.capture.release()
            raise ValueError(
                f"Invalid video properties for {task.video_path}: "
                f"fps={self.fps}, frames={self.expected_frames}, "
                f"size={self.width}x{self.height}"
            )

        task.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_jsonl = task.output_jsonl.with_suffix(
            task.output_jsonl.suffix + ".tmp"
        )
        self.handle = self.temporary_jsonl.open("w", encoding="utf-8")

        self.writer = None
        self.temporary_visualization: Path | None = None
        if task.visualization_path is not None:
            task.visualization_path.parent.mkdir(parents=True, exist_ok=True)
            self.temporary_visualization = _temporary_video_path(
                task.visualization_path
            )
            self.writer = cv2.VideoWriter(
                str(self.temporary_visualization),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            if not self.writer.isOpened():
                self.handle.close()
                self.capture.release()
                self.temporary_jsonl.unlink(missing_ok=True)
                raise ValueError(
                    "Could not create visualization: "
                    f"{self.temporary_visualization}"
                )

        self.frame_count = 0
        self.frames_with_person = 0
        self.multi_detection_frames = 0
        self.untracked_detection_frames = 0
        self.track_frame_counts: Counter[int] = Counter()
        self.confidences: list[float] = []
        self.area_ratios: list[float] = []
        self.local_track_ids: dict[int, int] = {}

    @property
    def active(self) -> bool:
        return self.frame_count < self.expected_frames

    def read_frame(self) -> np.ndarray:
        success, frame = self.capture.read()
        if not success:
            raise ValueError(
                f"Video ended at frame {self.frame_count}, expected "
                f"{self.expected_frames}: {self.task.video_path}"
            )
        return frame

    def _local_track_id(self, source_track_id: int) -> int:
        if source_track_id not in self.local_track_ids:
            self.local_track_ids[source_track_id] = (
                len(self.local_track_ids) + 1
            )
        return self.local_track_ids[source_track_id]

    def _detections_from_result(
        self,
        result,
        frame: np.ndarray,
    ) -> list[dict[str, object]]:
        boxes = result.boxes
        if boxes is None:
            return []

        detection_boxes = boxes.cpu().numpy()
        tracks = self.tracker.update(detection_boxes, frame)
        values: list[tuple[list[float], float, int | None]] = []
        if len(tracks) > 0:
            for track in tracks:
                source_track_id = int(track[4])
                values.append(
                    (
                        [float(value) for value in track[:4]],
                        float(track[5]),
                        self._local_track_id(source_track_id),
                    )
                )
        else:
            xyxy_values = np.asarray(detection_boxes.xyxy).tolist()
            confidence_values = np.asarray(detection_boxes.conf).tolist()
            values.extend(
                (
                    [float(value) for value in xyxy],
                    float(confidence),
                    None,
                )
                for xyxy, confidence in zip(
                    xyxy_values,
                    confidence_values,
                    strict=True,
                )
            )

        detections: list[dict[str, object]] = []
        for xyxy, confidence, track_id in values:
            left, top, right, bottom = xyxy
            area_ratio = max(0.0, right - left) * max(
                0.0, bottom - top
            ) / (self.width * self.height)
            detections.append(
                {
                    "track_id": track_id,
                    "bbox_xyxy": xyxy,
                    "bbox_xyxy_norm": [
                        left / self.width,
                        top / self.height,
                        right / self.width,
                        bottom / self.height,
                    ],
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                }
            )
        return detections

    def append_result(self, frame: np.ndarray, result) -> None:
        timestamp_s = self.frame_count / self.fps
        detections = self._detections_from_result(result, frame)
        if detections:
            self.frames_with_person += 1
        if len(detections) > 1:
            self.multi_detection_frames += 1

        frame_track_ids = {
            detection["track_id"]
            for detection in detections
            if detection["track_id"] is not None
        }
        self.track_frame_counts.update(frame_track_ids)
        if detections and not frame_track_ids:
            self.untracked_detection_frames += 1
        self.confidences.extend(
            float(detection["confidence"]) for detection in detections
        )
        self.area_ratios.extend(
            float(detection["area_ratio"]) for detection in detections
        )

        action_label = (
            label_at(self.task.annotations.segments, timestamp_s)
            if self.task.annotations is not None
            else None
        )
        row = {
            "path": self.task.relative_path,
            "frame_index": self.frame_count,
            "timestamp_s": timestamp_s,
            "width": self.width,
            "height": self.height,
            "action_label": action_label,
            "detections": detections,
        }
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if self.writer is not None:
            self.writer.write(
                _draw_frame(
                    frame.copy(),
                    detections,
                    timestamp_s=timestamp_s,
                    action_label=action_label,
                )
            )
        self.frame_count += 1

    def _close_resources(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.capture.release()
        if self.writer is not None:
            self.writer.release()

    def abort(self) -> None:
        self._close_resources()
        self.temporary_jsonl.unlink(missing_ok=True)
        if self.temporary_visualization is not None:
            self.temporary_visualization.unlink(missing_ok=True)

    def commit(self, elapsed_s: float) -> VideoTrackQuality:
        if self.frame_count != self.expected_frames:
            raise ValueError(
                f"Track/video frame mismatch for {self.task.video_path}: "
                f"{self.frame_count} != {self.expected_frames}"
            )
        self._close_resources()
        self.temporary_jsonl.replace(self.task.output_jsonl)
        if (
            self.task.visualization_path is not None
            and self.temporary_visualization is not None
        ):
            self.temporary_visualization.replace(
                self.task.visualization_path
            )
        return VideoTrackQuality(
            path=self.task.relative_path,
            expected_frames=self.expected_frames,
            processed_frames=self.frame_count,
            fps=self.fps,
            width=self.width,
            height=self.height,
            frames_with_person=self.frames_with_person,
            multi_detection_frames=self.multi_detection_frames,
            untracked_detection_frames=self.untracked_detection_frames,
            unique_track_ids=len(self.track_frame_counts),
            dominant_track_frames=max(
                self.track_frame_counts.values(),
                default=0,
            ),
            mean_person_confidence=(
                statistics.fmean(self.confidences)
                if self.confidences
                else 0.0
            ),
            median_bbox_area_ratio=(
                statistics.median(self.area_ratios)
                if self.area_ratios
                else 0.0
            ),
            elapsed_s=elapsed_s,
        )


def _read_active_frames(
    states: list[_VideoState],
    *,
    executor: ThreadPoolExecutor | None,
) -> tuple[list[_VideoState], list[np.ndarray]]:
    active_states = [state for state in states if state.active]
    if executor is None or len(active_states) <= 1:
        return active_states, [state.read_frame() for state in active_states]
    frames = list(
        executor.map(lambda state: state.read_frame(), active_states)
    )
    return active_states, frames


def track_video_batch(
    yolo_model,
    tasks: list[BatchVideoTask],
    *,
    config: TeacherConfig,
    decode_workers: int = 4,
    tracker_factory: TrackerFactory | None = None,
) -> list[VideoTrackQuality]:
    """Detect frames in cross-video batches and track each video independently."""

    if not tasks:
        return []
    if decode_workers <= 0:
        raise ValueError("decode_workers must be positive")

    factory = tracker_factory or _build_bytetracker
    trackers = [factory(config) for _ in tasks]
    states: list[_VideoState] = []
    executor = (
        ThreadPoolExecutor(max_workers=min(decode_workers, len(tasks)))
        if decode_workers > 1 and len(tasks) > 1
        else None
    )
    started_at = time.perf_counter()
    try:
        for task, tracker in zip(tasks, trackers, strict=True):
            states.append(_VideoState(task, tracker))
        yolo_model.model.end2end = config.end2end
        while any(state.active for state in states):
            active_states, frames = _read_active_frames(
                states,
                executor=executor,
            )
            results = yolo_model.predict(
                source=frames,
                stream=False,
                batch=len(frames),
                classes=[0],
                imgsz=list(config.image_size),
                conf=config.confidence,
                iou=config.iou,
                device=config.device,
                quantize=config.quantize,
                verbose=False,
                save=False,
            )
            if len(results) != len(active_states):
                raise RuntimeError(
                    "YOLO batch result count mismatch: "
                    f"{len(results)} != {len(active_states)}"
                )
            for state, frame, result in zip(
                active_states,
                frames,
                results,
                strict=True,
            ):
                state.append_result(frame, result)

        elapsed_s = time.perf_counter() - started_at
        amortized_elapsed_s = elapsed_s / len(states)
        return [
            state.commit(amortized_elapsed_s)
            for state in states
        ]
    except Exception:
        for state in states:
            state.abort()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
