"""Lightweight geometric and embedding association."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from edgefall_st.runtime.types import BBox, Detection, TrackSnapshot


def bbox_iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def cosine_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding dimensions must match")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 1.0
    similarity = sum(a * b for a, b in zip(first, second, strict=True))
    similarity /= first_norm * second_norm
    return 1.0 - max(-1.0, min(1.0, similarity))


def normalized_center_distance(first: BBox, second: BBox) -> float:
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y) / math.sqrt(2.0)


@dataclass(frozen=True)
class AssociationConfig:
    iou_weight: float = 0.55
    embedding_weight: float = 0.35
    center_weight: float = 0.10
    min_iou: float = 0.05
    max_embedding_distance: float = 0.55
    max_center_distance: float = 0.30
    max_cost: float = 0.80


def greedy_match(
    tracks: Sequence[TrackSnapshot],
    detections: Sequence[Detection],
    config: AssociationConfig = AssociationConfig(),
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return one-to-one (track index, detection index) assignments."""

    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            iou = bbox_iou(track.bbox, detection.bbox)
            embedding_distance = cosine_distance(
                track.embedding,
                detection.embedding,
            )
            center_distance = normalized_center_distance(
                track.bbox,
                detection.bbox,
            )
            if (
                iou < config.min_iou
                and embedding_distance > config.max_embedding_distance
            ):
                continue
            if center_distance > config.max_center_distance:
                continue
            cost = (
                config.iou_weight * (1.0 - iou)
                + config.embedding_weight * embedding_distance
                + config.center_weight * center_distance
            )
            if cost <= config.max_cost:
                candidates.append((cost, track_index, detection_index))

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for _, track_index, detection_index in sorted(candidates):
        if track_index in matched_tracks or detection_index in matched_detections:
            continue
        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)

    unmatched_tracks = [
        index for index in range(len(tracks)) if index not in matched_tracks
    ]
    unmatched_detections = [
        index
        for index in range(len(detections))
        if index not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections
