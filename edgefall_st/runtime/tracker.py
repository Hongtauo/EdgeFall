"""Small greedy multi-object tracker for EdgeFall-ST outputs."""

from __future__ import annotations

from dataclasses import dataclass

from edgefall_st.runtime.association import AssociationConfig, greedy_match
from edgefall_st.runtime.types import Detection, TrackSnapshot


@dataclass(frozen=True)
class TrackerConfig:
    max_misses: int = 5
    minimum_hits: int = 2
    embedding_momentum: float = 0.8
    association: AssociationConfig = AssociationConfig()


@dataclass
class _Track:
    track_id: int
    detection: Detection
    age: int = 1
    hits: int = 1
    misses: int = 0

    def snapshot(self) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=self.track_id,
            bbox=self.detection.bbox,
            person_score=self.detection.person_score,
            stage_probabilities=self.detection.stage_probabilities,
            onset_score=self.detection.onset_score,
            embedding=self.detection.embedding,
            age=self.age,
            hits=self.hits,
            misses=self.misses,
        )


class MultiObjectTracker:
    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        self.config = config
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: list[Detection],
    ) -> tuple[list[TrackSnapshot], list[int]]:
        snapshots = [track.snapshot() for track in self._tracks]
        matches, unmatched_tracks, unmatched_detections = greedy_match(
            snapshots,
            detections,
            self.config.association,
        )

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            momentum = self.config.embedding_momentum
            embedding = tuple(
                momentum * previous + (1.0 - momentum) * current
                for previous, current in zip(
                    track.detection.embedding,
                    detection.embedding,
                    strict=True,
                )
            )
            track.detection = Detection(
                bbox=detection.bbox,
                person_score=detection.person_score,
                stage_probabilities=detection.stage_probabilities,
                onset_score=detection.onset_score,
                embedding=embedding,
            )
            track.age += 1
            track.hits += 1
            track.misses = 0

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.age += 1
            track.misses += 1

        for detection_index in unmatched_detections:
            self._tracks.append(
                _Track(
                    track_id=self._next_track_id,
                    detection=detections[detection_index],
                )
            )
            self._next_track_id += 1

        expired = [
            track.track_id
            for track in self._tracks
            if track.misses > self.config.max_misses
        ]
        self._tracks = [
            track
            for track in self._tracks
            if track.misses <= self.config.max_misses
        ]
        visible = [
            track.snapshot()
            for track in self._tracks
            if track.misses == 0 and track.hits >= self.config.minimum_hits
        ]
        return visible, expired
