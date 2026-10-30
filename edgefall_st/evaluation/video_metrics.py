"""Manifest-backed video metrics shared by training and evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from edgefall_st.training.train import _binary_ranking_metrics


SCORE_FIELDS = ("score", "geometry_score", "global_score", "roi_score")


def manifest_domains(manifest_path: Path, *, split: str) -> tuple[str, ...]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        domains = {
            str(row["domain"])
            for row in csv.DictReader(handle)
            if row.get("split") == split
        }
    if not domains:
        raise ValueError(f"Video manifest selected no rows for split {split!r}")
    return tuple(sorted(domains))


def end_to_end_video_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    domain: str,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row.get("domain") == domain and row.get("split") == split
        ]
    if not selected:
        raise ValueError("Video manifest selected no rows for domain/split")
    expected = {str(row["path"]): bool(int(row["label"])) for row in selected}
    if len(expected) != len(selected):
        raise ValueError("Video manifest contains duplicate selected paths")
    maxima: dict[str, dict[str, float]] = {}
    available_fields = {"score"}
    for record in records:
        path = str(record["path"])
        if path not in expected:
            raise ValueError(f"Prediction path is absent from manifest: {path}")
        values = maxima.setdefault(path, {})
        for field in SCORE_FIELDS:
            value = record.get(field, "")
            if value == "" or value is None:
                continue
            available_fields.add(field)
            values[field] = max(values.get(field, 0.0), float(value))
    missing = [
        {"path": path, "label": int(label), "assigned_score": 0.0}
        for path, label in expected.items()
        if path not in maxima
    ]
    paths = list(expected)
    labels = [expected[path] for path in paths]
    result: dict[str, Any] = {
        "end_to_end_expected_videos": len(paths),
        "end_to_end_scored_videos": len(maxima),
        "end_to_end_missing_videos": len(missing),
        "end_to_end_missing_positive_videos": sum(
            int(row["label"]) for row in missing
        ),
        "end_to_end_missing_negative_videos": sum(
            1 - int(row["label"]) for row in missing
        ),
        "end_to_end_video_coverage": len(maxima) / len(paths),
    }
    for field in SCORE_FIELDS:
        if field not in available_fields:
            continue
        scores = [maxima.get(path, {}).get(field, 0.0) for path in paths]
        ranking = _binary_ranking_metrics(scores, labels)
        prefix = (
            "end_to_end_video"
            if field == "score"
            else f"end_to_end_{field.removesuffix('_score')}_video"
        )
        result.update(
            {f"{prefix}_{name}": value for name, value in ranking.items()}
        )
    return result, missing
