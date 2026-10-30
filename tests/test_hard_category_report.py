import csv
from pathlib import Path

import pytest

from edgefall_st.evaluation.hard_category_report import (
    _quantile,
    build_report,
    negative_categories,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_negative_categories_maps_semantic_actions() -> None:
    assert negative_categories("omnifall", "0|3|4") == {"sit"}
    assert negative_categories("omnifall", "5|6|10") == {
        "lie_down",
        "kneel_squat",
    }
    assert negative_categories("caucafall", "Pick up object") == {"bend_pick"}
    assert negative_categories("fall_video_dataset", "No_Fall") == {
        "ordinary_other"
    }


def test_quantile_uses_linear_interpolation() -> None:
    assert _quantile([0.0, 1.0], 0.9) == pytest.approx(0.9)


def test_build_report_uses_video_peak_and_missing_score_zero(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        manifest,
        [
            {
                "domain": "omnifall",
                "split": "test",
                "path": "sit/a",
                "label": 0,
                "label_name": "3|4",
            },
            {
                "domain": "omnifall",
                "split": "test",
                "path": "sit/missing",
                "label": 0,
                "label_name": "3",
            },
            {
                "domain": "omnifall",
                "split": "test",
                "path": "fall/a",
                "label": 1,
                "label_name": "1|2",
            },
        ],
    )
    _write_csv(
        baseline,
        [
            {"path": "sit/a", "score": 0.2},
            {"path": "sit/a", "score": 0.8},
        ],
    )
    _write_csv(
        candidate,
        [
            {"path": "sit/a", "score": 0.4},
            {"path": "fall/a", "score": 0.9},
        ],
    )

    summary, details = build_report(
        manifest,
        {"omnifall": baseline},
        {"omnifall": candidate},
        split="test",
    )

    sit = next(row for row in summary if row["category"] == "sit")
    assert sit["count"] == 2
    assert sit["baseline_mean"] == pytest.approx(0.4)
    assert sit["candidate_mean"] == pytest.approx(0.2)
    positive = next(
        row for row in summary if row["category"] == "positive_fall"
    )
    assert positive["candidate_mean"] == pytest.approx(0.9)
    assert details[1]["baseline_score"] == 0.0
