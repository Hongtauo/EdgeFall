import random
from pathlib import Path

from edgefall_st.data.multidomain import _cauca_rows, build_parser
from external_eval.caucafall.common import (
    ACTIVITY_LABELS,
    LIGHTING_BY_SUBJECT,
    CaucaVideo,
)


def _records() -> list[CaucaVideo]:
    return [
        CaucaVideo(
            sample_id=f"Subject.{subject_id}/{activity}",
            subject_id=subject_id,
            activity=activity,
            label=label,
            lighting=LIGHTING_BY_SUBJECT[subject_id],
            video_path=Path(f"Subject.{subject_id}/{activity}/video.avi"),
            frames=(),
        )
        for subject_id in range(1, 11)
        for activity, label in ACTIVITY_LABELS.items()
    ]


def test_cauca_protocol_uses_all_training_subject_videos() -> None:
    rows = _cauca_rows(
        _records(),
        train_count=60,
        randomizer=random.Random(20260727),
    )

    train = [row for row in rows if row.split == "train"]
    val = [row for row in rows if row.split == "val"]
    test = [row for row in rows if row.split == "test"]

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert {row.subject_id for row in train} == {"1", "2", "5", "6", "7", "9"}
    assert {row.subject_id for row in val} == {"3", "10"}
    assert {row.subject_id for row in test} == {"4", "8"}
    for activity in ACTIVITY_LABELS:
        assert sum(row.label_name == activity for row in train) == 6


def test_cauca60_defaults_do_not_overwrite_historical_manifest() -> None:
    args = build_parser().parse_args([])

    assert args.cauca_train_count == 60
    assert args.output == Path(
        "artifacts/multidomain_cauca60/split_manifest.csv"
    )
