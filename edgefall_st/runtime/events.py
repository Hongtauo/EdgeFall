"""Per-track fall event state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from edgefall_st.runtime.types import EventMessage, TrackSnapshot


class EventState(str, Enum):
    NORMAL = "normal"
    SUSPECTED = "suspected_fall"
    ALARMED = "alarmed"


@dataclass(frozen=True)
class EventConfig:
    falling_threshold: float = 0.55
    fallen_threshold: float = 0.60
    onset_threshold: float = 0.60
    confirmation_windows: int = 2
    recovery_windows: int = 3


@dataclass
class _TrackEvent:
    state: EventState = EventState.NORMAL
    start_s: float = 0.0
    last_positive_s: float = 0.0
    positive_windows: int = 0
    normal_windows: int = 0
    peak_falling: float = 0.0
    peak_fallen: float = 0.0
    peak_onset: float = 0.0

    @property
    def confidence(self) -> float:
        persistence = min(1.0, self.positive_windows / 4.0)
        score = (
            0.35 * self.peak_falling
            + 0.30 * self.peak_fallen
            + 0.25 * self.peak_onset
            + 0.10 * persistence
        )
        return max(0.0, min(1.0, score))


class FallEventManager:
    def __init__(self, config: EventConfig = EventConfig()) -> None:
        self.config = config
        self._events: dict[int, _TrackEvent] = {}

    def reset(self) -> None:
        self._events.clear()

    def update(
        self,
        timestamp_s: float,
        tracks: list[TrackSnapshot],
        expired_track_ids: list[int] | None = None,
    ) -> list[EventMessage]:
        messages: list[EventMessage] = []
        for track in tracks:
            event = self._events.setdefault(track.track_id, _TrackEvent())
            _, falling, fallen = track.stage_probabilities
            positive = (
                falling >= self.config.falling_threshold
                or fallen >= self.config.fallen_threshold
                or track.onset_score >= self.config.onset_threshold
            )
            if positive:
                if event.state is EventState.NORMAL:
                    event.state = EventState.SUSPECTED
                    event.start_s = timestamp_s
                event.positive_windows += 1
                event.normal_windows = 0
                event.last_positive_s = timestamp_s
                event.peak_falling = max(event.peak_falling, falling)
                event.peak_fallen = max(event.peak_fallen, fallen)
                event.peak_onset = max(event.peak_onset, track.onset_score)
                if (
                    event.state is EventState.SUSPECTED
                    and event.positive_windows >= self.config.confirmation_windows
                ):
                    event.state = EventState.ALARMED
                    messages.append(
                        EventMessage(
                            kind="alarm",
                            track_id=track.track_id,
                            start_s=event.start_s,
                            end_s=timestamp_s,
                            confidence=event.confidence,
                        )
                    )
            else:
                event.normal_windows += 1
                if (
                    event.state is EventState.SUSPECTED
                    and event.normal_windows >= self.config.recovery_windows
                ):
                    self._events[track.track_id] = _TrackEvent()
                elif (
                    event.state is EventState.ALARMED
                    and event.normal_windows >= self.config.recovery_windows
                ):
                    messages.append(
                        EventMessage(
                            kind="closed",
                            track_id=track.track_id,
                            start_s=event.start_s,
                            end_s=event.last_positive_s,
                            confidence=event.confidence,
                        )
                    )
                    self._events[track.track_id] = _TrackEvent()

        for track_id in expired_track_ids or []:
            message = self.close_track(track_id, timestamp_s)
            if message is not None:
                messages.append(message)
        return messages

    def close_track(
        self,
        track_id: int,
        timestamp_s: float,
    ) -> EventMessage | None:
        event = self._events.pop(track_id, None)
        if event is None or event.state is not EventState.ALARMED:
            return None
        return EventMessage(
            kind="closed",
            track_id=track_id,
            start_s=event.start_s,
            end_s=max(event.last_positive_s, timestamp_s),
            confidence=event.confidence,
        )
