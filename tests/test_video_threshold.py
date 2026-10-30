from __future__ import annotations

import csv
from pathlib import Path

from edgefall_st.evaluation.video_threshold import calibrate


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_threshold_is_selected_from_validation_only(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        ["domain", "split", "path", "label"],
        [
            {"domain": "a", "split": "val", "path": "vp", "label": 1},
            {"domain": "a", "split": "val", "path": "vn", "label": 0},
            {"domain": "a", "split": "test", "path": "tp", "label": 1},
            {"domain": "a", "split": "test", "path": "tn", "label": 0},
        ],
    )
    validation = tmp_path / "validation.csv"
    _write_csv(
        validation,
        ["path", "score"],
        [
            {"path": "vp", "score": 0.8},
            {"path": "vp", "score": 0.7},
            {"path": "vn", "score": 0.3},
        ],
    )
    test = tmp_path / "test.csv"
    _write_csv(
        test,
        ["path", "score"],
        [
            {"path": "tp", "score": 0.75},
            {"path": "tn", "score": 0.79},
        ],
    )
    result = calibrate(
        manifest_path=manifest,
        validation_predictions=[("a", validation)],
        test_predictions=[("a", test)],
    )
    assert result["selection"]["threshold"] == 0.8
    assert result["validation"]["pooled"]["f1"] == 1.0
    assert result["test"]["pooled"]["f1"] == 0.0
