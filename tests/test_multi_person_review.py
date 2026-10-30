from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edgefall_st.data.multi_person_review import build_supervision


REVIEW_FIELDS = (
    "path",
    "auto_group",
    "primary_track_ids",
    "secondary_track_ids",
    "manual_is_multi",
    "manual_primary_correct",
    "manual_notes",
)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _review_row(
    path: str,
    *,
    primary: str = "1",
    secondary: str = "2",
    is_multi: str = "yes",
    primary_correct: str = "yes",
) -> dict[str, str]:
    return {
        "path": path,
        "auto_group": "spatially_distinct",
        "primary_track_ids": primary,
        "secondary_track_ids": secondary,
        "manual_is_multi": is_multi,
        "manual_primary_correct": primary_correct,
        "manual_notes": "",
    }


def test_builds_positive_and_negative_track_supervision(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    decisions = tmp_path / "decisions.csv"
    _write_csv(
        review,
        REVIEW_FIELDS,
        [
            _review_row(
                "fall/verified",
                primary="1|2",
                secondary="2|3",
            ),
            _review_row(
                "fallen/corrected",
                primary="4",
                secondary="5",
                primary_correct="no",
            ),
            _review_row("sitting/negative", primary="7", secondary="8|9"),
            _review_row(
                "walking/not_multi",
                primary="10",
                secondary="11",
                is_multi="no",
            ),
        ],
    )
    _write_csv(
        decisions,
        ("path", "decision", "target_track_ids", "reason"),
        [
            {
                "path": "fallen/corrected",
                "decision": "use",
                "target_track_ids": "13",
                "reason": "manual correction",
            }
        ],
    )

    by_path = {
        record.path: record
        for record in build_supervision(review, decisions)
    }
    verified = by_path["fall/verified"]
    assert verified.target_track_ids == "1|2"
    assert verified.negative_track_ids == "3"
    corrected = by_path["fallen/corrected"]
    assert corrected.target_track_ids == "13"
    assert corrected.negative_track_ids == "4|5"
    negative = by_path["sitting/negative"]
    assert negative.negative_track_ids == "7|8|9"
    assert negative.action_target_track_ids == "7"
    not_multi = by_path["walking/not_multi"]
    assert not_multi.negative_track_ids == "10"
    assert not_multi.secondary_track_ids == "11"


def test_excludes_only_decided_multi_positive(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    decisions = tmp_path / "decisions.csv"
    _write_csv(
        review,
        REVIEW_FIELDS,
        [_review_row("fall/excluded", primary_correct="uncertain")],
    )
    _write_csv(
        decisions,
        ("path", "decision", "target_track_ids", "reason"),
        [
            {
                "path": "fall/excluded",
                "decision": "exclude",
                "target_track_ids": "",
                "reason": "unresolved",
            }
        ],
    )
    record = build_supervision(review, decisions)[0]
    assert record.use_in_training == 0
    assert record.use_in_person_supervision == 0
    assert record.disposition == "excluded_unresolved_positive"


def test_rejects_unresolved_multi_positive(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    decisions = tmp_path / "decisions.csv"
    _write_csv(
        review,
        REVIEW_FIELDS,
        [_review_row("fallen/unresolved", primary_correct="no")],
    )
    _write_csv(
        decisions,
        ("path", "decision", "target_track_ids", "reason"),
        [],
    )
    with pytest.raises(ValueError, match="requires a decision"):
        build_supervision(review, decisions)


def test_repository_review_decisions_are_complete() -> None:
    records = build_supervision(
        Path("annotations/omnifall/train_multi_person_manual_review.csv"),
        Path("annotations/omnifall/train_multi_person_decisions.csv"),
    )
    assert len(records) == 1603
    assert sum(not record.use_in_training for record in records) == 11
    assert sum(
        record.manual_is_multi == "yes"
        and record.video_fall_label == 1
        and record.use_in_training
        for record in records
    ) == 237
    assert sum(
        record.manual_is_multi == "yes" and record.video_fall_label == 0
        for record in records
    ) == 1010
