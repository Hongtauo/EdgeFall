from __future__ import annotations

import csv
from pathlib import Path

from edgefall_st.evaluation.failure_audit import audit


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(scores: dict[str, list[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, values in scores.items():
        for index, score in enumerate(values):
            rows.append(
                {
                    "path": path,
                    "window_index": index,
                    "end_time_s": float(index),
                    "stage_label": 0,
                    "action_label": 9,
                    "score": score,
                    "geometry_score": score - 0.1,
                    "global_score": score,
                    "roi_score": score,
                    "global_attention": 0.4,
                    "roi_attention": 0.6,
                }
            )
    return rows


def test_audit_reports_version_and_fusion_transitions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        ["domain", "split", "path", "label", "label_name", "video_path"],
        [
            {
                "domain": "d",
                "split": "test",
                "path": "fall/p1",
                "label": 1,
                "label_name": "fall",
                "video_path": "",
            },
            {
                "domain": "d",
                "split": "test",
                "path": "fall/p2",
                "label": 1,
                "label_name": "fall",
                "video_path": "",
            },
            {
                "domain": "d",
                "split": "test",
                "path": "other/n1",
                "label": 0,
                "label_name": "other",
                "video_path": "",
            },
            {
                "domain": "d",
                "split": "test",
                "path": "other/n2",
                "label": 0,
                "label_name": "other",
                "video_path": "",
            },
        ],
    )
    fields = [
        "path",
        "window_index",
        "end_time_s",
        "stage_label",
        "action_label",
        "score",
        "geometry_score",
        "global_score",
        "roi_score",
        "global_attention",
        "roi_attention",
    ]
    baseline = tmp_path / "baseline.csv"
    _write_csv(
        baseline,
        fields,
        _prediction_rows(
            {
                "fall/p1": [0.9],
                "fall/p2": [0.8],
                "other/n1": [0.4],
                "other/n2": [0.3],
            }
        ),
    )
    candidate = tmp_path / "candidate.csv"
    candidate_rows = _prediction_rows(
        {
            "fall/p1": [0.9],
            "fall/p2": [0.4],
            "other/n1": [0.8],
            "other/n2": [0.3],
        }
    )
    for row in candidate_rows:
        if row["path"] == "other/n1":
            row["geometry_score"] = 0.4
    _write_csv(candidate, fields, candidate_rows)

    result = audit(
        manifest_path=manifest,
        split="test",
        baseline_predictions=[("d", baseline)],
        candidate_predictions=[("d", candidate)],
        threshold=0.6,
        output_dir=tmp_path / "audit",
        review_per_category=2,
        render_panels=False,
        video_roots=[],
        actor_target_roots=[],
        online_domain=None,
        online_videos_path=None,
        online_frames_path=None,
    )

    locked = result["domains"]["d"]["locked_threshold"]
    assert locked["baseline"]["tp"] == 2
    assert locked["candidate"]["tp"] == 1
    assert locked["candidate"]["fp"] == 1
    assert locked["transitions"]["baseline_only_correct"] == 2
    assert locked["fusion_crossings_among_errors"] == {
        "aligned_low": 1,
        "fusion_raised": 1,
    }
    assert (tmp_path / "audit" / "failure_cases.csv").is_file()
    assert (tmp_path / "audit" / "review_manifest.csv").is_file()
    assert (tmp_path / "audit" / "summary.json").is_file()
