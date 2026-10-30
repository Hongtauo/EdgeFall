from __future__ import annotations

import pytest

from edgefall_st.data.annotations import (
    AmbiguousAnnotationError,
    Segment,
    clean_annotations,
    label_at,
)


def segment(
    path: str,
    label: int,
    start: float,
    end: float,
    row: int,
) -> Segment:
    return Segment(
        path=path,
        label=label,
        start=start,
        end=end,
        metadata={"age_group": "adult"},
        source_row=row,
    )


def test_clean_annotations_drops_and_merges() -> None:
    rows = [
        segment("fall/a", 1, 0.0, 1.0, 2),
        segment("fall/a", 1, 0.0, 1.0, 3),
        segment("fall/a", 1, 0.5, 1.5, 4),
        segment("fall/a", 2, 1.51, 5.0, 5),
        segment("fall/a", 2, 5.0, 5.0, 6),
    ]

    videos, report = clean_annotations(rows, boundary_fps=30.0)

    assert len(videos) == 1
    assert len(videos[0].segments) == 2
    assert videos[0].segments[0].start == 0.0
    assert videos[0].segments[0].end == pytest.approx(1.505)
    assert videos[0].segments[1].start == pytest.approx(1.505)
    assert report.dropped_duplicates == 1
    assert report.dropped_invalid == 1
    assert report.merged_same_label == 1
    assert report.snapped_boundaries == 1


def test_large_gap_is_preserved_as_unlabeled_time() -> None:
    rows = [
        segment("other/a", 8, 0.0, 1.0, 2),
        segment("other/a", 15, 1.1, 2.0, 3),
    ]

    videos, report = clean_annotations(rows, boundary_fps=30.0)

    assert report.large_gaps == 1
    assert label_at(videos[0].segments, 1.05) is None


def test_left_closed_right_open_semantics() -> None:
    rows = [
        segment("fall/a", 1, 0.0, 1.0, 2),
        segment("fall/a", 2, 1.0, 2.0, 3),
    ]
    videos, _ = clean_annotations(rows)

    assert label_at(videos[0].segments, 0.0) == 1
    assert label_at(videos[0].segments, 0.999) == 1
    assert label_at(videos[0].segments, 1.0) == 2
    assert label_at(videos[0].segments, 2.0) is None


def test_conflicting_large_overlap_is_not_silently_resolved() -> None:
    rows = [
        segment("fall/a", 1, 0.0, 2.0, 2),
        segment("fall/a", 4, 1.0, 3.0, 3),
    ]
    videos, report = clean_annotations(rows)

    assert report.large_overlaps == 1
    with pytest.raises(AmbiguousAnnotationError):
        label_at(videos[0].segments, 1.5)
