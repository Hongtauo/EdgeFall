from __future__ import annotations

from edgefall_st.data.annotations import Segment, VideoAnnotations
from edgefall_st.data.windows import build_fall_events, build_windows


def make_fall_video() -> VideoAnnotations:
    metadata = {"age_group": "adult"}
    segments = (
        Segment("fall/a", 8, 0.0, 1.0, metadata),
        Segment("fall/a", 1, 1.0, 2.0, metadata),
        Segment("fall/a", 2, 2.0, 4.0, metadata),
        Segment("fall/a", 7, 4.0, 5.0, metadata),
    )
    return VideoAnnotations("fall/a", segments)


def test_windows_are_causal_and_use_final_frame_label() -> None:
    windows = list(
        build_windows(
            make_fall_video(),
            effective_fps=10.0,
            window_length=32,
            stride=4,
        )
    )

    first = windows[0]
    assert first.end_frame == 0
    assert first.left_padding == 31
    assert first.stage_label == 0

    at_fall = next(window for window in windows if window.end_time_s == 1.2)
    assert at_fall.action_label == 1
    assert at_fall.stage_label == 1
    assert at_fall.start_time_s == 0.0


def test_fall_and_following_fallen_form_one_event() -> None:
    events = build_fall_events(make_fall_video())

    assert len(events) == 1
    assert events[0].start_s == 1.0
    assert events[0].falling_end_s == 2.0
    assert events[0].end_s == 4.0
    assert events[0].includes_fallen


def test_short_fall_segment_is_not_skipped_by_regular_stride() -> None:
    metadata = {"age_group": "adult"}
    video = VideoAnnotations(
        "fall/short",
        (
            Segment("fall/short", 8, 0.0, 0.15, metadata),
            Segment("fall/short", 1, 0.15, 0.25, metadata),
            Segment("fall/short", 8, 0.25, 0.8, metadata),
        ),
    )

    windows = list(
        build_windows(
            video,
            effective_fps=10.0,
            window_length=32,
            stride=4,
        )
    )

    fall_windows = [window for window in windows if window.action_label == 1]
    assert [window.end_frame for window in fall_windows] == [2]
