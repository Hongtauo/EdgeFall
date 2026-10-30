from __future__ import annotations

from pathlib import Path

import pytest

from external_eval.caucafall.common import (
    DatasetLayoutError,
    discover_videos,
    read_annotation,
    select_balanced_records,
)


def _make_activity(
    root: Path,
    subject: int,
    activity: str,
    labels: list[int],
    *,
    alias_index: int | None = None,
) -> None:
    activity_root = root / f"Subject.{subject}" / activity
    activity_root.mkdir(parents=True, exist_ok=True)
    (activity_root / f"{activity.replace(' ', '')}S{subject}.avi").write_bytes(
        b"avi"
    )
    (activity_root / "classes.txt").write_text(
        "nofall\nfall\n", encoding="utf-8"
    )
    for index, label in enumerate(labels, start=1):
        stem = f"frame{subject:02d}{index:04d}"
        (activity_root / f"{stem}.png").write_bytes(b"png")
        annotation_stem = stem
        if alias_index == index:
            annotation_stem += "a"
        (activity_root / f"{annotation_stem}.txt").write_text(
            f"{label} 0.5 0.5 0.4 0.6\n",
            encoding="utf-8",
        )


def test_read_annotation_converts_yolo_box(tmp_path: Path) -> None:
    path = tmp_path / "frame.txt"
    path.write_text("1 0.5 0.4 0.2 0.4\n", encoding="utf-8")

    target, box = read_annotation(path)

    assert target is True
    assert box == pytest.approx((0.4, 0.2, 0.6, 0.6))


def test_read_annotation_clips_author_box_to_image(tmp_path: Path) -> None:
    path = tmp_path / "frame.txt"
    path.write_text("0 0.95 0.5 0.2 0.4\n", encoding="utf-8")

    target, box = read_annotation(path)

    assert target is False
    assert box == pytest.approx((0.85, 0.3, 1.0, 0.7))


def test_discover_videos_pairs_unique_numeric_alias(tmp_path: Path) -> None:
    _make_activity(
        tmp_path,
        1,
        "Fall forward",
        [0, 1],
        alias_index=2,
    )
    _make_activity(tmp_path, 1, "Walk", [0, 0])

    records = discover_videos(tmp_path)

    assert len(records) == 2
    fall = next(record for record in records if record.label)
    assert fall.fall_frames == 1
    assert fall.alias_pairs == 1
    assert fall.missing_annotations == 0


def test_directory_and_frame_labels_must_agree(tmp_path: Path) -> None:
    _make_activity(tmp_path, 1, "Walk", [0, 1])

    with pytest.raises(DatasetLayoutError, match="disagree"):
        discover_videos(tmp_path)


def test_balanced_limit_is_per_binary_label(tmp_path: Path) -> None:
    _make_activity(tmp_path, 1, "Fall forward", [0, 1])
    _make_activity(tmp_path, 1, "Fall left", [0, 1])
    _make_activity(tmp_path, 1, "Walk", [0, 0])
    _make_activity(tmp_path, 1, "Hop", [0, 0])
    records = discover_videos(tmp_path)

    selected = select_balanced_records(records, 1)

    assert len(selected) == 2
    assert sum(record.label for record in selected) == 1
