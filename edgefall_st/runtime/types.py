"""Runtime data structures independent of the neural network framework."""

from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    person_score: float
    stage_probabilities: tuple[float, float, float]
    onset_score: float
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    bbox: BBox
    person_score: float
    stage_probabilities: tuple[float, float, float]
    onset_score: float
    embedding: tuple[float, ...]
    age: int
    hits: int
    misses: int


@dataclass(frozen=True)
class EventMessage:
    kind: str
    track_id: int
    start_s: float
    end_s: float
    confidence: float
