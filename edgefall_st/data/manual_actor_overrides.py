"""Build reviewed OmniFall actor overrides and a base-cache selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from edgefall_st.data.multi_person_review import load_decisions
from edgefall_st.pseudo.actor_targets import (
    load_track_frames,
    select_primary_actor,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _relative_jsonl(root: Path, sample_path: str) -> Path:
    relative = Path(*sample_path.split("/"))
    return root / relative.parent / f"{relative.name}.jsonl"


def build_overrides(
    *,
    decisions_path: Path,
    tracks_root: Path,
    windows_path: Path,
    base_cache_index: Path,
    output_root: Path,
) -> Path:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    decisions = load_decisions(decisions_path)
    corrected = {
        path: decision.target_track_ids[0]
        for path, decision in decisions.items()
        if decision.decision == "use"
    }
    excluded = {
        path for path, decision in decisions.items() if decision.decision == "exclude"
    }
    if any(
        len(decision.target_track_ids) != 1
        for decision in decisions.values()
        if decision.decision == "use"
    ):
        raise ValueError("Manual actor overrides require exactly one target track ID")

    quality_rows: list[dict[str, Any]] = []
    for sample_path, track_id in sorted(corrected.items()):
        frames = load_track_frames(_relative_jsonl(tracks_root, sample_path))
        targets, quality = select_primary_actor(
            frames,
            required_track_id=track_id,
        )
        _write_jsonl(
            _relative_jsonl(output_root / "targets", sample_path),
            targets,
        )
        quality_rows.append(
            {
                "path": sample_path,
                "required_track_id": track_id,
                "frame_count": quality.frame_count,
                "actor_coverage": quality.actor_coverage,
                "selected_source_track_ids": list(
                    quality.selected_source_track_ids
                ),
                "source_track_switches": quality.source_track_switches,
                "confidence_level": quality.confidence_level,
                "action_loss_weight": quality.action_loss_weight,
            }
        )

    base_rows = _read_jsonl(base_cache_index)
    original_window_indices = {
        (str(row["path"]), int(row["end_frame"])): int(row["window_index"])
        for row in base_rows
    }
    corrected_windows = []
    for row in _read_jsonl(windows_path):
        sample_path = str(row.get("path"))
        if sample_path not in corrected:
            continue
        key = (sample_path, int(row["end_frame"]))
        if key not in original_window_indices:
            raise ValueError(f"Window is absent from base cache: {key}")
        corrected_windows.append(
            {**row, "window_index": original_window_indices[key]}
        )
    corrected_window_paths = {
        str(row["path"]) for row in corrected_windows
    }
    if corrected_window_paths != set(corrected):
        missing = sorted(set(corrected).difference(corrected_window_paths))
        raise ValueError(f"Corrected paths are absent from windows: {missing}")
    corrected_window_count = _write_jsonl(
        output_root / "windows.jsonl",
        corrected_windows,
    )

    removed_from_base = set(corrected) | excluded
    base_paths = {str(row["path"]) for row in base_rows}
    missing_base = sorted(removed_from_base.difference(base_paths))
    if missing_base:
        raise ValueError(f"Reviewed paths are absent from base cache: {missing_base}")
    selection_rows = (
        {
            "cache_index": index,
            "selected": True,
            "path": str(row["path"]),
        }
        for index, row in enumerate(base_rows)
        if str(row["path"]) not in removed_from_base
    )
    selected_base_windows = _write_jsonl(
        output_root / "base_cache_selection.jsonl",
        selection_rows,
    )
    path_window_counts = Counter(str(row["path"]) for row in base_rows)
    summary = {
        "decisions": str(decisions_path),
        "tracks_root": str(tracks_root),
        "windows": str(windows_path),
        "base_cache_index": str(base_cache_index),
        "corrected_paths": dict(sorted(corrected.items())),
        "excluded_paths": sorted(excluded),
        "base_windows": len(base_rows),
        "selected_base_windows": selected_base_windows,
        "corrected_windows": corrected_window_count,
        "excluded_windows": sum(path_window_counts[path] for path in excluded),
        "final_training_windows": selected_base_windows + corrected_window_count,
        "quality": quality_rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--decisions",
        dest="decisions_path",
        type=Path,
        default=Path("annotations/omnifall/train_multi_person_decisions.csv"),
    )
    parser.add_argument(
        "--tracks-root",
        type=Path,
        default=Path("artifacts/pseudo_labels/train_yolo26s_nms/tracks"),
    )
    parser.add_argument(
        "--windows",
        dest="windows_path",
        type=Path,
        default=Path("artifacts/omnifall_v2/windows_train.jsonl"),
    )
    parser.add_argument(
        "--base-cache-index",
        type=Path,
        default=Path("artifacts/frozen_features/yolo26s/train/index.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/manual_actor_overrides/omnifall_train"),
    )
    return parser


def main() -> None:
    summary = build_overrides(**vars(build_parser().parse_args()))
    print(f"manual_actor_overrides={summary}", flush=True)


if __name__ == "__main__":
    main()
