from __future__ import annotations

from collections import Counter

from edgefall_st.data.annotations import Segment, VideoAnnotations
from edgefall_st.data.split import SplitRatios, stratified_video_split


def make_video(index: int) -> VideoAnnotations:
    label = index % 4
    metadata = {
        "age_group": f"age_{index % 3}",
        "environment_category": f"env_{index % 2}",
    }
    segment = Segment(
        path=f"video/{index:03d}",
        label=label,
        start=0.0,
        end=5.0,
        metadata=metadata,
    )
    return VideoAnnotations(segment.path, (segment,))


def test_split_is_exact_deterministic_and_video_level() -> None:
    videos = [make_video(index) for index in range(100)]

    first = stratified_video_split(
        videos,
        ratios=SplitRatios(0.7, 0.1, 0.2),
        seed=17,
    )
    second = stratified_video_split(
        videos,
        ratios=SplitRatios(0.7, 0.1, 0.2),
        seed=17,
    )

    assert first == second
    assert len(first) == len(videos)
    assert Counter(first.values()) == {"train": 70, "val": 10, "test": 20}
    assert set(first) == {video.path for video in videos}


def test_invalid_ratios_are_rejected() -> None:
    videos = [make_video(index) for index in range(10)]

    try:
        stratified_video_split(
            videos,
            ratios=SplitRatios(0.8, 0.1, 0.2),
        )
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("Invalid split ratios should fail")
