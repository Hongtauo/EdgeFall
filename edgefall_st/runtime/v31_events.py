"""Causal event state machine for V3 bbox-classifier predictions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Literal


class V31EventState(str, Enum):
    NORMAL = "normal"
    FALLING = "falling"
    FALLEN = "fallen"
    ALARMED = "alarmed"


@dataclass(frozen=True)
class V31StateMachineConfig:
    confirmation_windows: int = 2
    maximum_gap_s: float = 0.6
    transition_window_s: float = 2.0
    alarm_threshold: float = 0.5
    release_threshold: float = 0.25
    recovery_windows: int = 3
    use_stage_transition: bool = True

    def __post_init__(self) -> None:
        if self.confirmation_windows < 2:
            raise ValueError("confirmation_windows must be at least 2")
        if self.recovery_windows <= 0:
            raise ValueError("recovery_windows must be positive")
        if not isfinite(self.maximum_gap_s) or self.maximum_gap_s <= 0:
            raise ValueError("maximum_gap_s must be positive and finite")
        if not isfinite(self.transition_window_s) or self.transition_window_s <= 0:
            raise ValueError("transition_window_s must be positive and finite")
        if not 0.0 <= self.release_threshold <= self.alarm_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= release <= alarm <= 1"
            )


@dataclass(frozen=True)
class V31WindowPrediction:
    timestamp_s: float
    fall_score: float
    stage_probabilities: tuple[float, float, float]
    onset_score: float

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative and finite")
        for name, value in (
            ("fall_score", self.fall_score),
            ("onset_score", self.onset_score),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if len(self.stage_probabilities) != 3:
            raise ValueError("stage_probabilities must contain three values")
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.stage_probabilities
        ):
            raise ValueError("stage probabilities must be in [0, 1]")
        if abs(sum(self.stage_probabilities) - 1.0) > 1e-3:
            raise ValueError("stage probabilities must sum to one")


@dataclass(frozen=True)
class V31EventMessage:
    kind: Literal["alarm", "closed"]
    start_s: float
    end_s: float
    alarm_s: float
    confidence: float
    peak_fall_score: float


@dataclass(frozen=True)
class V31StateMachineUpdate:
    state: V31EventState
    persistence_score: float
    progression_score: float
    confirmation_score: float
    messages: tuple[V31EventMessage, ...]


class V31FallStateMachine:
    """Convert causal V3 window predictions into one-shot fall alarms.

    The confirmation score is independent of the configured alarm threshold.
    It is therefore suitable for validation-set threshold scans and P90/P95.
    """

    def __init__(
        self,
        config: V31StateMachineConfig = V31StateMachineConfig(),
    ) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = V31EventState.NORMAL
        self._recent: deque[V31WindowPrediction] = deque(
            maxlen=self.config.confirmation_windows
        )
        self._transitions: deque[tuple[float, float]] = deque()
        self._last_timestamp_s: float | None = None
        self._event_start_s = 0.0
        self._alarm_s = 0.0
        self._last_active_s = 0.0
        self._peak_confirmation = 0.0
        self._peak_fall = 0.0
        self._recovery_count = 0

    def _reset_event(self) -> None:
        self.state = V31EventState.NORMAL
        self._event_start_s = 0.0
        self._alarm_s = 0.0
        self._last_active_s = 0.0
        self._peak_confirmation = 0.0
        self._peak_fall = 0.0
        self._recovery_count = 0

    def _reset_evidence(self) -> None:
        self._recent.clear()
        self._transitions.clear()

    def _close(self, end_s: float) -> V31EventMessage:
        message = V31EventMessage(
            kind="closed",
            start_s=self._event_start_s,
            end_s=max(end_s, self._last_active_s),
            alarm_s=self._alarm_s,
            confidence=self._peak_confirmation,
            peak_fall_score=self._peak_fall,
        )
        self._reset_event()
        return message

    def _evidence_scores(
        self,
        prediction: V31WindowPrediction,
    ) -> tuple[float, float, float]:
        if (
            self._last_timestamp_s is not None
            and prediction.timestamp_s < self._last_timestamp_s
        ):
            raise ValueError("prediction timestamps must be non-decreasing")

        self._recent.append(prediction)
        persistence_score = 0.0
        if len(self._recent) == self.config.confirmation_windows:
            recent = tuple(self._recent)
            gaps_are_valid = all(
                current.timestamp_s - previous.timestamp_s
                <= self.config.maximum_gap_s
                for previous, current in zip(recent, recent[1:])
            )
            if gaps_are_valid:
                persistence_score = min(item.fall_score for item in recent)

        cutoff = prediction.timestamp_s - self.config.transition_window_s
        while self._transitions and self._transitions[0][0] < cutoff:
            self._transitions.popleft()
        prior_transition = max(
            (score for _, score in self._transitions),
            default=0.0,
        )
        _, falling, fallen = prediction.stage_probabilities
        fallen_evidence = min(prediction.fall_score, fallen)
        progression_score = (
            min(prior_transition, fallen_evidence)
            if self.config.use_stage_transition
            else 0.0
        )
        transition_evidence = min(
            prediction.fall_score,
            max(falling, prediction.onset_score),
        )
        self._transitions.append(
            (prediction.timestamp_s, transition_evidence)
        )
        confirmation_score = max(persistence_score, progression_score)
        return persistence_score, progression_score, confirmation_score

    def update(
        self,
        prediction: V31WindowPrediction,
    ) -> V31StateMachineUpdate:
        messages: list[V31EventMessage] = []
        if (
            self._last_timestamp_s is not None
            and prediction.timestamp_s - self._last_timestamp_s
            > self.config.transition_window_s
        ):
            if self.state is V31EventState.ALARMED:
                messages.append(self._close(self._last_active_s))
            else:
                self._reset_event()
            self._reset_evidence()

        persistence, progression, confirmation = self._evidence_scores(
            prediction
        )
        self._last_timestamp_s = prediction.timestamp_s
        _, _, fallen = prediction.stage_probabilities
        fallen_evidence = min(prediction.fall_score, fallen)
        alarm_threshold = self.config.alarm_threshold

        if (
            self.state is V31EventState.NORMAL
            and prediction.fall_score >= alarm_threshold
        ):
            self.state = V31EventState.FALLING
            self._event_start_s = prediction.timestamp_s
            self._last_active_s = prediction.timestamp_s
            self._peak_fall = prediction.fall_score

        if self.state in (V31EventState.FALLING, V31EventState.FALLEN):
            self._peak_fall = max(self._peak_fall, prediction.fall_score)
            self._peak_confirmation = max(
                self._peak_confirmation,
                confirmation,
            )
            if fallen_evidence >= alarm_threshold:
                self.state = V31EventState.FALLEN
            if confirmation >= alarm_threshold:
                self.state = V31EventState.FALLEN
                self._alarm_s = prediction.timestamp_s
                self._last_active_s = prediction.timestamp_s
                self._recovery_count = 0
                messages.append(
                    V31EventMessage(
                        kind="alarm",
                        start_s=self._event_start_s,
                        end_s=prediction.timestamp_s,
                        alarm_s=prediction.timestamp_s,
                        confidence=self._peak_confirmation,
                        peak_fall_score=self._peak_fall,
                    )
                )
                self.state = V31EventState.ALARMED
            elif prediction.fall_score >= self.config.release_threshold:
                self._last_active_s = prediction.timestamp_s
                self._recovery_count = 0
            else:
                self._recovery_count += 1
                if self._recovery_count >= self.config.recovery_windows:
                    self._reset_event()

        elif self.state is V31EventState.ALARMED:
            self._peak_fall = max(self._peak_fall, prediction.fall_score)
            self._peak_confirmation = max(
                self._peak_confirmation,
                confirmation,
            )
            active = max(prediction.fall_score, confirmation) >= (
                self.config.release_threshold
            )
            if active:
                self._last_active_s = prediction.timestamp_s
                self._recovery_count = 0
            else:
                self._recovery_count += 1
                if self._recovery_count >= self.config.recovery_windows:
                    messages.append(self._close(self._last_active_s))

        return V31StateMachineUpdate(
            state=self.state,
            persistence_score=persistence,
            progression_score=progression,
            confirmation_score=confirmation,
            messages=tuple(messages),
        )

    def flush(self) -> tuple[V31EventMessage, ...]:
        if self.state is not V31EventState.ALARMED:
            self._reset_event()
            self._reset_evidence()
            self._last_timestamp_s = None
            return ()
        message = self._close(self._last_active_s)
        self._reset_evidence()
        self._last_timestamp_s = None
        return (message,)


def v31_confirmation_scores(
    predictions: Iterable[V31WindowPrediction],
    config: V31StateMachineConfig = V31StateMachineConfig(),
) -> tuple[float, ...]:
    machine = V31FallStateMachine(config)
    return tuple(
        machine.update(prediction).confirmation_score
        for prediction in predictions
    )


def v31_sequence_score(
    predictions: Iterable[V31WindowPrediction],
    config: V31StateMachineConfig = V31StateMachineConfig(),
) -> float:
    return max(v31_confirmation_scores(predictions, config), default=0.0)
