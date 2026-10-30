"""Build primary-actor supervision from saved person tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from edgefall_st.data.labels import (
    ACTION_LABELS,
    FALL_ACTION_ID,
    action_to_stage,
)

ALGORITHM_VERSION = 2
HORIZONTAL_ACTION_IDS = frozenset((2, 6, 14))
VERTICAL_ACTION_IDS = frozenset((0, 7, 8, 15))


@dataclass(frozen=True)
class ActorSelectionConfig:
    """Weights and confidence gates for primary-actor path selection."""

    node_area_weight: float = 1.50
    node_confidence_weight: float = 0.45
    node_center_weight: float = 0.15
    node_track_coverage_weight: float = 0.10
    node_posture_weight: float = 0.15
    node_action_motion_weight: float = 3.50
    transition_iou_weight: float = 0.65
    transition_center_weight: float = 0.60
    transition_scale_weight: float = 0.20
    transition_same_track_weight: float = 0.25
    transition_switch_penalty: float = 0.10
    high_score_threshold: float = 0.78
    medium_score_threshold: float = 0.60
    high_min_coverage: float = 0.90
    medium_min_coverage: float = 0.65
    high_max_ambiguity_ratio: float = 0.50
    medium_action_weight: float = 0.35

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ActorSelectionQuality:
    path: str
    frame_count: int
    assigned_frames: int
    actor_coverage: float
    mean_prominence: float
    mean_margin: float
    mean_continuity: float
    mean_detection_confidence: float
    mean_action_consistency: float
    mean_action_motion: float
    ambiguous_frame_ratio: float
    source_track_switches: int
    selected_source_track_ids: tuple[int, ...]
    mean_path_score: float
    confidence_score: float
    confidence_level: str
    action_loss_weight: float

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["selected_source_track_ids"] = list(
            self.selected_source_track_ids
        )
        return value


@dataclass(frozen=True)
class _Candidates:
    detection_indices: np.ndarray
    boxes: np.ndarray
    areas: np.ndarray
    centers: np.ndarray
    confidences: np.ndarray
    track_ids: np.ndarray
    prominence: np.ndarray
    posture: np.ndarray
    action_motion: np.ndarray
    node_scores: np.ndarray


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def _posture_score(action_label: int | None, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float64)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    denominator = np.maximum(widths + heights, 1e-6)
    if action_label in HORIZONTAL_ACTION_IDS:
        return widths / denominator
    if action_label in VERTICAL_ACTION_IDS:
        return heights / denominator
    return np.full((len(boxes),), 0.5, dtype=np.float64)


def _valid_box(detection: dict[str, object]) -> np.ndarray | None:
    values = detection.get("bbox_xyxy_norm")
    if not isinstance(values, list) or len(values) != 4:
        return None
    box = np.asarray(values, dtype=np.float64)
    if not np.isfinite(box).all():
        return None
    box = np.clip(box, 0.0, 1.0)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _track_coverages(
    frames: Sequence[dict[str, object]],
) -> dict[int, float]:
    counts: dict[int, int] = {}
    for frame in frames:
        seen: set[int] = set()
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            if track_id is not None:
                seen.add(int(track_id))
        for track_id in seen:
            counts[track_id] = counts.get(track_id, 0) + 1
    denominator = max(1, len(frames))
    return {
        track_id: count / denominator for track_id, count in counts.items()
    }


def _track_prominences(
    frames: Sequence[dict[str, object]],
) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for frame in frames:
        frame_values: list[tuple[int, float]] = []
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            box = _valid_box(detection)
            if track_id is None or box is None:
                continue
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            frame_values.append((int(track_id), area))
        if not frame_values:
            continue
        maximum = max(area for _, area in frame_values)
        for track_id, area in frame_values:
            values.setdefault(track_id, []).append(
                math.sqrt(area / max(maximum, 1e-9))
            )
    return {
        track_id: statistics.median(prominences)
        for track_id, prominences in values.items()
    }


def _motion_compatibility(
    action_label: int | None,
    first_box: np.ndarray,
    last_box: np.ndarray,
) -> float:
    first_width = float(first_box[2] - first_box[0])
    first_height = float(first_box[3] - first_box[1])
    last_width = float(last_box[2] - last_box[0])
    last_height = float(last_box[3] - last_box[1])
    mean_height = max(0.05, 0.5 * (first_height + last_height))
    first_center = np.asarray(
        (
            (first_box[0] + first_box[2]) * 0.5,
            (first_box[1] + first_box[3]) * 0.5,
        )
    )
    last_center = np.asarray(
        (
            (last_box[0] + last_box[2]) * 0.5,
            (last_box[1] + last_box[3]) * 0.5,
        )
    )
    delta_center = last_center - first_center
    delta_y = float(delta_center[1]) / mean_height
    delta_height = (last_height - first_height) / mean_height
    first_aspect = first_width / max(first_height, 1e-6)
    last_aspect = last_width / max(last_height, 1e-6)
    delta_aspect = math.log(
        max(last_aspect, 1e-6) / max(first_aspect, 1e-6)
    )
    movement = float(np.linalg.norm(delta_center)) / mean_height

    if action_label == 7:
        signal = 0.75 * (-delta_y) + 0.25 * delta_height
    elif action_label in (3, 10, 12):
        signal = 0.75 * delta_y - 0.25 * delta_height
    elif action_label in (1, 5):
        signal = 0.60 * delta_y + 0.40 * delta_aspect
    elif action_label in (0, 14, 15):
        return float(np.clip(1.0 - math.exp(-1.5 * movement), 0.0, 1.0))
    else:
        return 0.5
    return float(np.clip(0.5 + 0.5 * math.tanh(2.5 * signal), 0.0, 1.0))


def _action_motion_scores(
    frames: Sequence[dict[str, object]],
) -> list[dict[int, float]]:
    scores: list[dict[int, float]] = [{} for _ in frames]
    segment_start = 0
    while segment_start < len(frames):
        action_value = frames[segment_start].get("action_label")
        action_label = None if action_value is None else int(action_value)
        segment_end = segment_start + 1
        while (
            segment_end < len(frames)
            and frames[segment_end].get("action_label") == action_value
        ):
            segment_end += 1

        observations: dict[int, list[tuple[int, np.ndarray]]] = {}
        for frame_position in range(segment_start, segment_end):
            frame = frames[frame_position]
            for detection in frame.get("detections", []):
                track_id = detection.get("track_id")
                box = _valid_box(detection)
                if track_id is None or box is None:
                    continue
                observations.setdefault(int(track_id), []).append(
                    (frame_position, box)
                )

        segment_length = segment_end - segment_start
        minimum_observations = max(3, math.ceil(0.20 * segment_length))
        for track_id, values in observations.items():
            if len(values) < minimum_observations:
                score = 0.5
            else:
                edge_count = max(1, math.ceil(0.20 * len(values)))
                first_box = np.median(
                    np.stack([box for _, box in values[:edge_count]]),
                    axis=0,
                )
                last_box = np.median(
                    np.stack([box for _, box in values[-edge_count:]]),
                    axis=0,
                )
                raw_score = _motion_compatibility(
                    action_label,
                    first_box,
                    last_box,
                )
                reliability = min(
                    1.0,
                    len(values) / max(1.0, 0.50 * segment_length),
                )
                score = 0.5 + reliability * (raw_score - 0.5)
            for frame_position, _ in values:
                scores[frame_position][track_id] = score
        segment_start = segment_end

    video_actor_prior: dict[int, float] = {}
    for frame_scores in scores:
        for track_id, score in frame_scores.items():
            video_actor_prior[track_id] = max(
                video_actor_prior.get(track_id, 0.5),
                score,
            )
    for frame_scores in scores:
        for track_id, score in tuple(frame_scores.items()):
            frame_scores[track_id] = max(
                score,
                video_actor_prior.get(track_id, 0.5),
            )
    return scores


def _build_candidates(
    frame: dict[str, object],
    track_coverages: dict[int, float],
    track_prominences: dict[int, float],
    action_motion_scores: dict[int, float],
    config: ActorSelectionConfig,
) -> _Candidates:
    indices: list[int] = []
    boxes: list[np.ndarray] = []
    confidences: list[float] = []
    track_ids: list[int] = []
    for index, detection in enumerate(frame.get("detections", [])):
        box = _valid_box(detection)
        if box is None:
            continue
        confidence = float(detection.get("confidence", 0.0))
        if not math.isfinite(confidence):
            confidence = 0.0
        track_id = detection.get("track_id")
        indices.append(index)
        boxes.append(box)
        confidences.append(float(np.clip(confidence, 0.0, 1.0)))
        track_ids.append(-1 if track_id is None else int(track_id))

    if not boxes:
        empty = np.empty((0,), dtype=np.float64)
        return _Candidates(
            detection_indices=np.empty((0,), dtype=np.int64),
            boxes=np.empty((0, 4), dtype=np.float64),
            areas=empty,
            centers=np.empty((0, 2), dtype=np.float64),
            confidences=empty,
            track_ids=np.empty((0,), dtype=np.int64),
            prominence=empty,
            posture=empty,
            action_motion=empty,
            node_scores=empty,
        )

    box_values = np.stack(boxes)
    widths = box_values[:, 2] - box_values[:, 0]
    heights = box_values[:, 3] - box_values[:, 1]
    areas = widths * heights
    centers = np.stack(
        (
            (box_values[:, 0] + box_values[:, 2]) * 0.5,
            (box_values[:, 1] + box_values[:, 3]) * 0.5,
        ),
        axis=1,
    )
    prominence = np.sqrt(areas / max(float(areas.max()), 1e-9))
    center_distance_sq = np.square(centers - 0.5).sum(axis=1)
    center_score = np.exp(-2.0 * center_distance_sq)
    track_id_values = np.asarray(track_ids, dtype=np.int64)
    coverage_values = np.asarray(
        [
            track_coverages.get(track_id, 0.0)
            if track_id >= 0
            else 0.0
            for track_id in track_ids
        ],
        dtype=np.float64,
    )
    action_label = frame.get("action_label")
    posture = _posture_score(
        None if action_label is None else int(action_label),
        box_values,
    )
    raw_action_motion = np.asarray(
        [
            action_motion_scores.get(track_id, 0.5)
            if track_id >= 0
            else 0.5
            for track_id in track_ids
        ],
        dtype=np.float64,
    )
    prominence_prior = np.asarray(
        [
            track_prominences.get(track_id, prominence[index])
            if track_id >= 0
            else prominence[index]
            for index, track_id in enumerate(track_ids)
        ],
        dtype=np.float64,
    )
    action_motion = 0.5 + prominence_prior * (raw_action_motion - 0.5)
    confidence_values = np.asarray(confidences, dtype=np.float64)
    node_scores = (
        config.node_area_weight * prominence
        + config.node_confidence_weight * confidence_values
        + config.node_center_weight * center_score
        + config.node_track_coverage_weight * coverage_values
        + config.node_posture_weight * posture
        + config.node_action_motion_weight * action_motion
    )
    return _Candidates(
        detection_indices=np.asarray(indices, dtype=np.int64),
        boxes=box_values,
        areas=areas,
        centers=centers,
        confidences=confidence_values,
        track_ids=track_id_values,
        prominence=prominence,
        posture=posture,
        action_motion=action_motion,
        node_scores=node_scores,
    )


def _transition_components(
    previous: _Candidates,
    current: _Candidates,
    frame_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous_boxes = previous.boxes[:, None, :]
    current_boxes = current.boxes[None, :, :]
    intersection_left_top = np.maximum(
        previous_boxes[..., :2],
        current_boxes[..., :2],
    )
    intersection_right_bottom = np.minimum(
        previous_boxes[..., 2:],
        current_boxes[..., 2:],
    )
    intersection_size = np.maximum(
        0.0,
        intersection_right_bottom - intersection_left_top,
    )
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = (
        previous.areas[:, None]
        + current.areas[None, :]
        - intersection
    )
    iou = intersection / np.maximum(union, 1e-9)

    center_distance = np.linalg.norm(
        previous.centers[:, None, :] - current.centers[None, :, :],
        axis=2,
    )
    object_scale = 0.35 * (
        np.sqrt(previous.areas[:, None])
        + np.sqrt(current.areas[None, :])
    )
    motion_scale = object_scale + 0.06 * max(1, frame_gap)
    center_similarity = np.exp(
        -center_distance / np.maximum(motion_scale, 1e-6)
    )

    log_scale_change = np.abs(
        np.log(
            np.maximum(current.areas[None, :], 1e-9)
            / np.maximum(previous.areas[:, None], 1e-9)
        )
    )
    scale_similarity = np.exp(
        -log_scale_change / (1.0 + 0.10 * max(0, frame_gap - 1))
    )
    return iou, center_similarity, scale_similarity


def _transition_scores(
    previous: _Candidates,
    current: _Candidates,
    frame_gap: int,
    config: ActorSelectionConfig,
) -> np.ndarray:
    iou, center_similarity, scale_similarity = _transition_components(
        previous,
        current,
        frame_gap,
    )
    previous_ids = previous.track_ids[:, None]
    current_ids = current.track_ids[None, :]
    both_tracked = (previous_ids >= 0) & (current_ids >= 0)
    same_track = both_tracked & (previous_ids == current_ids)
    switched_track = both_tracked & (previous_ids != current_ids)
    gap_penalty = min(0.30, 0.03 * max(0, frame_gap - 1))
    return (
        config.transition_iou_weight * iou
        + config.transition_center_weight * center_similarity
        + config.transition_scale_weight * scale_similarity
        + config.transition_same_track_weight * same_track
        - config.transition_switch_penalty * switched_track
        - gap_penalty
    )


def _path_confidence(
    *,
    frames: Sequence[dict[str, object]],
    candidates: Sequence[_Candidates],
    selected_positions: Sequence[int | None],
    mean_path_score: float,
    config: ActorSelectionConfig,
) -> ActorSelectionQuality:
    prominences: list[float] = []
    margins: list[float] = []
    confidences: list[float] = []
    posture_scores: list[float] = []
    action_motion_scores: list[float] = []
    continuities: list[float] = []
    selected_track_ids: list[int] = []
    source_track_switches = 0
    ambiguous_multi_frames = 0
    multi_frames = 0
    previous: tuple[int, _Candidates, int] | None = None

    for frame_position, selected in enumerate(selected_positions):
        if selected is None:
            continue
        current = candidates[frame_position]
        selected_area = float(current.areas[selected])
        prominences.append(float(current.prominence[selected]))
        confidences.append(float(current.confidences[selected]))
        posture_scores.append(float(current.posture[selected]))
        action_motion_scores.append(float(current.action_motion[selected]))

        if len(current.areas) == 1:
            margin = 1.0
        else:
            multi_frames += 1
            alternative = float(
                np.max(np.delete(current.areas, selected))
            )
            margin = max(0.0, 1.0 - alternative / selected_area)
            if margin < 0.15:
                ambiguous_multi_frames += 1
        margins.append(margin)

        track_id = int(current.track_ids[selected])
        if track_id >= 0:
            selected_track_ids.append(track_id)
        if previous is not None:
            previous_frame, previous_candidates, previous_selected = previous
            frame_gap = frame_position - previous_frame
            iou, center_similarity, scale_similarity = (
                _transition_components(
                    previous_candidates,
                    current,
                    frame_gap,
                )
            )
            continuities.append(
                0.35 * float(iou[previous_selected, selected])
                + 0.45
                * float(center_similarity[previous_selected, selected])
                + 0.20
                * float(scale_similarity[previous_selected, selected])
            )
            previous_track_id = int(
                previous_candidates.track_ids[previous_selected]
            )
            if (
                previous_track_id >= 0
                and track_id >= 0
                and previous_track_id != track_id
            ):
                source_track_switches += 1
        previous = (frame_position, current, selected)

    frame_count = len(frames)
    assigned_frames = len(prominences)
    coverage = assigned_frames / frame_count if frame_count else 0.0
    mean_prominence = _mean(prominences)
    mean_margin = _mean(margins)
    mean_continuity = _mean(continuities, default=coverage)
    mean_confidence = _mean(confidences)
    mean_action_consistency = _mean(posture_scores, default=0.5)
    mean_action_motion = _mean(action_motion_scores, default=0.5)
    ambiguity_ratio = (
        ambiguous_multi_frames / multi_frames if multi_frames else 0.0
    )
    confidence_score = (
        0.28 * coverage
        + 0.20 * mean_prominence
        + 0.15 * mean_margin
        + 0.15 * mean_continuity
        + 0.08 * mean_confidence
        + 0.04 * mean_action_consistency
        + 0.10 * mean_action_motion
    )
    if (
        coverage >= config.high_min_coverage
        and confidence_score >= config.high_score_threshold
        and ambiguity_ratio <= config.high_max_ambiguity_ratio
    ):
        confidence_level = "high"
        action_loss_weight = 1.0
    elif (
        coverage >= config.medium_min_coverage
        and confidence_score >= config.medium_score_threshold
    ):
        confidence_level = "medium"
        action_loss_weight = config.medium_action_weight
    else:
        confidence_level = "low"
        action_loss_weight = 0.0

    path = str(frames[0].get("path", "")) if frames else ""
    return ActorSelectionQuality(
        path=path,
        frame_count=frame_count,
        assigned_frames=assigned_frames,
        actor_coverage=coverage,
        mean_prominence=mean_prominence,
        mean_margin=mean_margin,
        mean_continuity=mean_continuity,
        mean_detection_confidence=mean_confidence,
        mean_action_consistency=mean_action_consistency,
        mean_action_motion=mean_action_motion,
        ambiguous_frame_ratio=ambiguity_ratio,
        source_track_switches=source_track_switches,
        selected_source_track_ids=tuple(sorted(set(selected_track_ids))),
        mean_path_score=mean_path_score,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        action_loss_weight=action_loss_weight,
    )


def select_primary_actor(
    frames: Sequence[dict[str, object]],
    *,
    config: ActorSelectionConfig | None = None,
    required_track_id: int | None = None,
) -> tuple[list[dict[str, object]], ActorSelectionQuality]:
    """Select one temporally coherent primary-actor box per frame."""

    settings = config or ActorSelectionConfig()
    if not frames:
        raise ValueError("frames must not be empty")
    if required_track_id is not None and required_track_id < 0:
        raise ValueError("required_track_id cannot be negative")
    path = frames[0].get("path")
    for expected_index, frame in enumerate(frames):
        if frame.get("path") != path:
            raise ValueError("All track rows must belong to the same video")
        if int(frame.get("frame_index", -1)) != expected_index:
            raise ValueError(
                f"Non-contiguous frame index: expected {expected_index}, "
                f"got {frame.get('frame_index')}"
            )
        action_label = frame.get("action_label")
        if action_label is not None and int(action_label) not in ACTION_LABELS:
            raise ValueError(f"Unknown action label: {action_label}")

    track_coverages = _track_coverages(frames)
    track_prominences = _track_prominences(frames)
    action_motion_scores = _action_motion_scores(frames)
    candidates = [
        _build_candidates(
            frame,
            track_coverages,
            track_prominences,
            action_motion_scores[frame_position],
            settings,
        )
        for frame_position, frame in enumerate(frames)
    ]
    if required_track_id is not None and not any(
        np.any(current.track_ids == required_track_id) for current in candidates
    ):
        raise ValueError(
            f"Required track ID {required_track_id} is absent from {path}"
        )
    state_frames: list[int] = []
    state_candidates: list[_Candidates] = []
    state_scores: list[np.ndarray] = []
    backpointers: list[np.ndarray | None] = []
    previous_frame = -1
    for frame_position, current in enumerate(candidates):
        if len(current.boxes) == 0:
            continue
        node_scores = current.node_scores.copy()
        if required_track_id is not None:
            required = current.track_ids == required_track_id
            if np.any(required):
                node_scores[~required] = -np.inf
        if not state_scores:
            scores = node_scores
            backpointer = None
        else:
            previous_candidates = state_candidates[-1]
            transition = _transition_scores(
                previous_candidates,
                current,
                frame_position - previous_frame,
                settings,
            )
            combined = state_scores[-1][:, None] + transition
            backpointer = np.argmax(combined, axis=0)
            scores = node_scores + combined[
                backpointer,
                np.arange(len(current.boxes)),
            ]
        state_frames.append(frame_position)
        state_candidates.append(current)
        state_scores.append(scores)
        backpointers.append(backpointer)
        previous_frame = frame_position

    selected_positions: list[int | None] = [None] * len(frames)
    if state_scores:
        selected = int(np.argmax(state_scores[-1]))
        for state_index in range(len(state_scores) - 1, -1, -1):
            frame_position = state_frames[state_index]
            selected_positions[frame_position] = selected
            backpointer = backpointers[state_index]
            if backpointer is not None:
                selected = int(backpointer[selected])
        mean_path_score = float(np.max(state_scores[-1])) / len(
            state_scores
        )
    else:
        mean_path_score = 0.0

    quality = _path_confidence(
        frames=frames,
        candidates=candidates,
        selected_positions=selected_positions,
        mean_path_score=mean_path_score,
        config=settings,
    )

    targets: list[dict[str, object]] = []
    previous_action: int | None = None
    for frame, current, selected in zip(
        frames,
        candidates,
        selected_positions,
        strict=True,
    ):
        action_value = frame.get("action_label")
        action_label = (
            None if action_value is None else int(action_value)
        )
        onset_target = int(
            action_label == FALL_ACTION_ID
            and previous_action != FALL_ACTION_ID
        )
        detection_index = None
        source_track_id = None
        bbox = None
        confidence = None
        area_ratio = None
        actor_identity = None
        if selected is not None:
            detection_index = int(current.detection_indices[selected])
            detection = frame["detections"][detection_index]
            source_track_id = detection.get("track_id")
            bbox = [
                float(value)
                for value in detection["bbox_xyxy_norm"]
            ]
            confidence = float(detection.get("confidence", 0.0))
            area_ratio = float(detection.get("area_ratio", 0.0))
            actor_identity = 0

        target_weight = (
            quality.action_loss_weight
            if detection_index is not None and action_label is not None
            else 0.0
        )
        targets.append(
            {
                "path": path,
                "frame_index": int(frame["frame_index"]),
                "timestamp_s": float(frame["timestamp_s"]),
                "action_label": action_label,
                "stage_label": (
                    action_to_stage(action_label)
                    if action_label is not None
                    else None
                ),
                "onset_target": onset_target,
                "actor_detection_index": detection_index,
                "actor_source_track_id": source_track_id,
                "actor_identity": actor_identity,
                "actor_bbox_xyxy_norm": bbox,
                "actor_confidence": confidence,
                "actor_area_ratio": area_ratio,
                "action_loss_weight": target_weight,
                "track_loss_weight": target_weight,
            }
        )
        previous_action = action_label
    return targets, quality


def load_track_frames(path: str | Path) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
    if not frames:
        raise ValueError(f"Track file contains no frames: {path}")
    return frames


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_jsonl_atomic(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _artifact_path(root: Path, video_path: str, suffix: str) -> Path:
    relative = Path(*video_path.split("/"))
    return root / relative.parent / f"{relative.name}{suffix}"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise ValueError(f"Manifest has no path column: {path}")
        return list(reader)


def _quality_manifest_row(
    split: str,
    quality: ActorSelectionQuality,
) -> dict[str, object]:
    return {
        "path": quality.path,
        "split": split,
        "confidence_level": quality.confidence_level,
        "confidence_score": f"{quality.confidence_score:.8f}",
        "action_loss_weight": f"{quality.action_loss_weight:.4f}",
        "actor_coverage": f"{quality.actor_coverage:.8f}",
        "mean_prominence": f"{quality.mean_prominence:.8f}",
        "mean_margin": f"{quality.mean_margin:.8f}",
        "mean_continuity": f"{quality.mean_continuity:.8f}",
        "mean_detection_confidence": (
            f"{quality.mean_detection_confidence:.8f}"
        ),
        "mean_action_consistency": (
            f"{quality.mean_action_consistency:.8f}"
        ),
        "mean_action_motion": f"{quality.mean_action_motion:.8f}",
        "ambiguous_frame_ratio": (
            f"{quality.ambiguous_frame_ratio:.8f}"
        ),
        "source_track_switches": quality.source_track_switches,
        "selected_source_track_ids": "|".join(
            str(track_id)
            for track_id in quality.selected_source_track_ids
        ),
        "frame_count": quality.frame_count,
        "assigned_frames": quality.assigned_frames,
    }


def _write_manifest_atomic(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else ["path", "split"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summarize(
    split: str,
    qualities: Sequence[ActorSelectionQuality],
) -> dict[str, object]:
    levels = {
        level: sum(
            quality.confidence_level == level for quality in qualities
        )
        for level in ("high", "medium", "low")
    }
    return {
        "split": split,
        "processed_videos": len(qualities),
        "processed_frames": sum(
            quality.frame_count for quality in qualities
        ),
        "assigned_frames": sum(
            quality.assigned_frames for quality in qualities
        ),
        "confidence_levels": levels,
        "mean_confidence_score": _mean(
            [quality.confidence_score for quality in qualities]
        ),
        "mean_actor_coverage": _mean(
            [quality.actor_coverage for quality in qualities]
        ),
        "mean_prominence": _mean(
            [quality.mean_prominence for quality in qualities]
        ),
        "mean_margin": _mean(
            [quality.mean_margin for quality in qualities]
        ),
        "mean_continuity": _mean(
            [quality.mean_continuity for quality in qualities]
        ),
        "mean_ambiguity_ratio": _mean(
            [quality.ambiguous_frame_ratio for quality in qualities]
        ),
        "mean_action_motion": _mean(
            [quality.mean_action_motion for quality in qualities]
        ),
        "mean_source_track_switches": _mean(
            [
                float(quality.source_track_switches)
                for quality in qualities
            ]
        ),
        "low_confidence_videos": [
            quality.path
            for quality in qualities
            if quality.confidence_level == "low"
        ],
        "medium_confidence_videos": [
            quality.path
            for quality in qualities
            if quality.confidence_level == "medium"
        ],
    }


def _load_quality(path: Path) -> ActorSelectionQuality:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["selected_source_track_ids"] = tuple(
        int(track_id)
        for track_id in value["selected_source_track_ids"]
    )
    return ActorSelectionQuality(**value)


def _check_run_config(
    path: Path,
    expected: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != expected and not overwrite:
        raise ValueError(
            f"Actor target config changed for {path.parent}. "
            "Use a new output root or pass --overwrite."
        )


def build_split_actor_targets(
    *,
    split: str,
    pseudo_root: Path,
    output_root: Path,
    pseudo_suffix: str = "yolo26x_nms",
    config: ActorSelectionConfig | None = None,
    workers: int = 8,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build actor references for one dataset split."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    settings = config or ActorSelectionConfig()
    if not pseudo_suffix or Path(pseudo_suffix).name != pseudo_suffix:
        raise ValueError("pseudo_suffix must be a non-empty directory name")
    source_root = pseudo_root / f"{split}_{pseudo_suffix}"
    source_manifest = source_root / "pilot_manifest.csv"
    manifest_rows = _read_manifest(source_manifest)
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    split_output = output_root / split
    run_config = {
        "algorithm_version": ALGORITHM_VERSION,
        "split": split,
        "source_root": str(source_root.resolve()),
        "selection": settings.as_dict(),
    }
    run_config_path = split_output / "run_config.json"
    _check_run_config(
        run_config_path,
        run_config,
        overwrite=overwrite,
    )
    _write_json_atomic(run_config_path, run_config)

    def process(row: dict[str, str]) -> ActorSelectionQuality:
        video_path = row["path"]
        track_path = _artifact_path(
            source_root / "tracks",
            video_path,
            ".jsonl",
        )
        target_path = _artifact_path(
            split_output / "targets",
            video_path,
            ".jsonl",
        )
        quality_path = _artifact_path(
            split_output / "quality",
            video_path,
            ".json",
        )
        if target_path.exists() and quality_path.exists() and not overwrite:
            return _load_quality(quality_path)
        frames = load_track_frames(track_path)
        targets, quality = select_primary_actor(
            frames,
            config=settings,
        )
        _write_jsonl_atomic(target_path, targets)
        _write_json_atomic(quality_path, quality.as_dict())
        return quality

    qualities: list[ActorSelectionQuality] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, quality in enumerate(
            executor.map(process, manifest_rows),
            start=1,
        ):
            qualities.append(quality)
            if index == 1 or index % 100 == 0 or index == len(manifest_rows):
                print(
                    f"[{split} {index}/{len(manifest_rows)}] "
                    f"{quality.path}: {quality.confidence_level} "
                    f"score={quality.confidence_score:.3f} "
                    f"coverage={quality.actor_coverage:.3f}"
                )

    manifest = [
        _quality_manifest_row(split, quality)
        for quality in qualities
    ]
    _write_manifest_atomic(
        split_output / "actor_manifest.csv",
        manifest,
    )
    summary = _summarize(split, qualities)
    summary["source_manifest"] = str(source_manifest.resolve())
    summary["selection_config"] = settings.as_dict()
    _write_json_atomic(split_output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
    )
    parser.add_argument(
        "--pseudo-root",
        type=Path,
        default=Path("artifacts/pseudo_labels"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/actor_targets_v1"),
    )
    parser.add_argument("--pseudo-suffix", default="yolo26x_nms")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    summaries = {
        split: build_split_actor_targets(
            split=split,
            pseudo_root=args.pseudo_root,
            output_root=args.output_root,
            pseudo_suffix=args.pseudo_suffix,
            workers=args.workers,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        for split in splits
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
