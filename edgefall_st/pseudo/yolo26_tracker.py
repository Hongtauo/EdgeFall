"""YOLO26 + ByteTrack offline teacher inference."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from edgefall_st.data.annotations import VideoAnnotations, label_at
from edgefall_st.data.labels import ACTION_LABELS
from edgefall_st.pseudo.quality import VideoTrackQuality


@dataclass(frozen=True)
class TeacherConfig:
    model: str = "yolo26x.pt"
    image_size: tuple[int, int] = (544, 960)
    confidence: float = 0.10
    iou: float = 0.30
    tracker: str = "bytetrack.yaml"
    device: str = "0"
    quantize: int | None = 16
    end2end: bool = False


def video_properties(video_path: str | Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video properties for {video_path}: "
            f"fps={fps}, frames={frame_count}, size={width}x{height}"
        )
    return fps, frame_count, width, height


def _draw_frame(
    frame,
    detections: list[dict[str, object]],
    *,
    timestamp_s: float,
    action_label: int | None,
    actor_detection_index: int | None = None,
):
    for detection_index, detection in enumerate(detections):
        left, top, right, bottom = (
            round(value) for value in detection["bbox_xyxy"]
        )
        track_id = detection["track_id"]
        confidence = detection["confidence"]
        is_actor = detection_index == actor_detection_index
        color = (
            (40, 40, 255)
            if is_actor
            else (
                (40, 220, 40)
                if track_id is not None
                else (0, 180, 255)
            )
        )
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            4 if is_actor else 2,
        )
        prefix = "ACTOR " if is_actor else ""
        cv2.putText(
            frame,
            f"{prefix}id={track_id} person={confidence:.2f}",
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    action_name = ACTION_LABELS.get(action_label, "unlabeled")
    cv2.putText(
        frame,
        f"t={timestamp_s:.3f}s action={action_name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def render_track_visualization(
    video_path: str | Path,
    tracks_jsonl: str | Path,
    output_path: str | Path,
    actor_targets_jsonl: str | Path | None = None,
) -> int:
    """Render saved track records without rerunning the teacher model."""

    fps, expected_frames, width, height = video_properties(video_path)
    actor_rows = None
    if actor_targets_jsonl is not None:
        actor_rows = []
        with Path(actor_targets_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    actor_rows.append(json.loads(line))
        if len(actor_rows) != expected_frames:
            raise ValueError(
                "Actor target/video frame mismatch for "
                f"{actor_targets_jsonl}: {len(actor_rows)} != "
                f"{expected_frames}"
            )
    capture = cv2.VideoCapture(str(video_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create visualization: {temporary}")

    rendered_frames = 0
    try:
        try:
            with Path(tracks_jsonl).open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    success, frame = capture.read()
                    if not success:
                        raise ValueError(
                            f"Video ended before track row {rendered_frames}: "
                            f"{video_path}"
                        )
                    if row["frame_index"] != rendered_frames:
                        raise ValueError(
                            f"Non-contiguous track rows in {tracks_jsonl}: "
                            f"expected {rendered_frames}, "
                            f"got {row['frame_index']}"
                        )
                    actor_detection_index = None
                    if actor_rows is not None:
                        actor_row = actor_rows[rendered_frames]
                        if actor_row["frame_index"] != rendered_frames:
                            raise ValueError(
                                "Non-contiguous actor target rows in "
                                f"{actor_targets_jsonl}: expected "
                                f"{rendered_frames}, got "
                                f"{actor_row['frame_index']}"
                            )
                        actor_detection_index = actor_row[
                            "actor_detection_index"
                        ]
                    writer.write(
                        _draw_frame(
                            frame,
                            row["detections"],
                            timestamp_s=row["timestamp_s"],
                            action_label=row["action_label"],
                            actor_detection_index=actor_detection_index,
                        )
                    )
                    rendered_frames += 1
            if rendered_frames != expected_frames:
                raise ValueError(
                    f"Track/video frame mismatch for {video_path}: "
                    f"{rendered_frames} != {expected_frames}"
                )
        finally:
            capture.release()
            writer.release()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    temporary.replace(output)
    return rendered_frames


def track_video(
    yolo_model,
    video_path: str | Path,
    output_jsonl: str | Path,
    *,
    relative_path: str,
    config: TeacherConfig,
    annotations: VideoAnnotations | None = None,
    visualization_path: str | Path | None = None,
) -> VideoTrackQuality:
    """Track every source frame and atomically write one JSONL record per frame."""

    fps, expected_frames, width, height = video_properties(video_path)
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    writer = None
    if visualization_path is not None:
        visualization = Path(visualization_path)
        visualization.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(visualization),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not create visualization: {visualization}")

    frame_count = 0
    frames_with_person = 0
    multi_detection_frames = 0
    untracked_detection_frames = 0
    track_frame_counts: Counter[int] = Counter()
    confidences: list[float] = []
    area_ratios: list[float] = []
    start_time = time.perf_counter()

    try:
        yolo_model.model.end2end = config.end2end
        results = yolo_model.track(
            source=str(video_path),
            stream=True,
            persist=False,
            classes=[0],
            tracker=config.tracker,
            imgsz=list(config.image_size),
            conf=config.confidence,
            iou=config.iou,
            device=config.device,
            quantize=config.quantize,
            verbose=False,
            save=False,
        )
        with temporary.open("w", encoding="utf-8") as handle:
            for frame_index, result in enumerate(results):
                timestamp_s = frame_index / fps
                detections: list[dict[str, object]] = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy_values = boxes.xyxy.detach().cpu().tolist()
                    confidence_values = boxes.conf.detach().cpu().tolist()
                    if boxes.id is None:
                        track_ids: list[int | None] = [None] * len(boxes)
                    else:
                        track_ids = [
                            int(value)
                            for value in boxes.id.detach().cpu().tolist()
                        ]
                    for xyxy, confidence, track_id in zip(
                        xyxy_values,
                        confidence_values,
                        track_ids,
                        strict=True,
                    ):
                        left, top, right, bottom = (
                            float(value) for value in xyxy
                        )
                        normalized = [
                            left / width,
                            top / height,
                            right / width,
                            bottom / height,
                        ]
                        area_ratio = max(0.0, right - left) * max(
                            0.0, bottom - top
                        ) / (width * height)
                        detection = {
                            "track_id": track_id,
                            "bbox_xyxy": [left, top, right, bottom],
                            "bbox_xyxy_norm": normalized,
                            "confidence": float(confidence),
                            "area_ratio": area_ratio,
                        }
                        detections.append(detection)
                        confidences.append(float(confidence))
                        area_ratios.append(area_ratio)

                if detections:
                    frames_with_person += 1
                if len(detections) > 1:
                    multi_detection_frames += 1
                frame_track_ids = {
                    detection["track_id"]
                    for detection in detections
                    if detection["track_id"] is not None
                }
                track_frame_counts.update(frame_track_ids)
                if detections and not frame_track_ids:
                    untracked_detection_frames += 1

                action_label = (
                    label_at(annotations.segments, timestamp_s)
                    if annotations is not None
                    else None
                )
                row = {
                    "path": relative_path,
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "width": width,
                    "height": height,
                    "action_label": action_label,
                    "detections": detections,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                if writer is not None:
                    frame = result.orig_img.copy()
                    writer.write(
                        _draw_frame(
                            frame,
                            detections,
                            timestamp_s=timestamp_s,
                            action_label=action_label,
                        )
                    )
                frame_count += 1
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.release()
        if temporary.exists():
            temporary.unlink()

    elapsed_s = time.perf_counter() - start_time
    return VideoTrackQuality(
        path=relative_path,
        expected_frames=expected_frames,
        processed_frames=frame_count,
        fps=fps,
        width=width,
        height=height,
        frames_with_person=frames_with_person,
        multi_detection_frames=multi_detection_frames,
        untracked_detection_frames=untracked_detection_frames,
        unique_track_ids=len(track_frame_counts),
        dominant_track_frames=max(track_frame_counts.values(), default=0),
        mean_person_confidence=(
            statistics.fmean(confidences) if confidences else 0.0
        ),
        median_bbox_area_ratio=(
            statistics.median(area_ratios) if area_ratios else 0.0
        ),
        elapsed_s=elapsed_s,
    )
