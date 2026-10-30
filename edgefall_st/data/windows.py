"""Causal frame-window and event label generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Sequence

from edgefall_st.data.annotations import Segment, VideoAnnotations, label_at
from edgefall_st.data.labels import FALL_ACTION_ID, FALLEN_ACTION_ID, action_to_stage


@dataclass(frozen=True)
class WindowLabel:
    path: str
    end_frame: int
    end_time_s: float
    start_time_s: float
    left_padding: int
    action_label: int
    stage_label: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FallEvent:
    path: str
    start_s: float
    end_s: float
    falling_end_s: float
    includes_fallen: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_windows(
    video: VideoAnnotations,
    *,
    effective_fps: float = 10.0,
    window_length: int = 32,
    stride: int = 4,
) -> Iterator[WindowLabel]:
    """Generate causal windows with regular and segment-aware endpoints."""

    if effective_fps <= 0:
        raise ValueError("effective_fps must be positive")
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")

    frame_count = max(1, math.ceil(video.duration * effective_fps))
    end_frames = set(range(0, frame_count, stride))
    for segment in video.segments:
        first_sample = max(
            0,
            math.ceil(segment.start * effective_fps - 1e-9),
        )
        last_sample = min(
            frame_count - 1,
            math.ceil(segment.end * effective_fps - 1e-9) - 1,
        )
        if first_sample > last_sample:
            continue
        end_frames.update(
            (first_sample, (first_sample + last_sample) // 2, last_sample)
        )

    for end_frame in sorted(end_frames):
        end_time_s = end_frame / effective_fps
        action_label = label_at(video.segments, end_time_s)
        if action_label is None:
            continue
        first_frame = end_frame - window_length + 1
        left_padding = max(0, -first_frame)
        start_time_s = max(0.0, first_frame / effective_fps)
        yield WindowLabel(
            path=video.path,
            end_frame=end_frame,
            end_time_s=end_time_s,
            start_time_s=start_time_s,
            left_padding=left_padding,
            action_label=action_label,
            stage_label=action_to_stage(action_label),
        )


def build_fall_events(
    video: VideoAnnotations,
    *,
    continuation_tolerance_s: float = 1.0 / 30.0,
) -> list[FallEvent]:
    """Build observable fall events from fall -> optional fallen transitions."""

    segments: Sequence[Segment] = video.segments
    events: list[FallEvent] = []
    for index, segment in enumerate(segments):
        if segment.label != FALL_ACTION_ID:
            continue
        event_end = segment.end
        includes_fallen = False
        if index + 1 < len(segments):
            following = segments[index + 1]
            gap = following.start - segment.end
            if (
                following.label == FALLEN_ACTION_ID
                and abs(gap) <= continuation_tolerance_s
            ):
                event_end = following.end
                includes_fallen = True
        events.append(
            FallEvent(
                path=video.path,
                start_s=segment.start,
                end_s=event_end,
                falling_end_s=segment.end,
                includes_fallen=includes_fallen,
            )
        )
    return events


def iter_all_windows(
    videos: Iterable[VideoAnnotations],
    *,
    effective_fps: float,
    window_length: int,
    stride: int,
) -> Iterator[WindowLabel]:
    for video in videos:
        yield from build_windows(
            video,
            effective_fps=effective_fps,
            window_length=window_length,
            stride=stride,
        )
