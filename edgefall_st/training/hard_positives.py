"""Load and validate training-only hard-positive anchor manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence


@dataclass(frozen=True)
class HardPositiveRegistry:
    anchors: Mapping[str, tuple[int, ...]]
    anchor_domains: Mapping[int, str]
    summary: Mapping[str, Any]

    @property
    def enabled(self) -> bool:
        return bool(self.anchors)


def empty_hard_positive_registry() -> HardPositiveRegistry:
    return HardPositiveRegistry(
        anchors={},
        anchor_domains={},
        summary={"candidate_count": 0, "video_count": 0},
    )


def load_hard_positive_registry(
    path: str | Path,
    windows: Sequence[dict[str, Any]],
    *,
    positive_paths: Collection[str],
) -> HardPositiveRegistry:
    manifest_path = Path(path)
    positives = {str(value) for value in positive_paths}
    key_to_index: dict[tuple[str, int], int] = {}
    for index, window in enumerate(windows):
        if "window_index" not in window:
            raise ValueError(
                "Training windows must expose window_index for hard positives"
            )
        key = (str(window["path"]), int(window["window_index"]))
        if key in key_to_index:
            raise ValueError(f"Duplicate selected training window: {key}")
        key_to_index[key] = index

    anchors: dict[str, list[int]] = defaultdict(list)
    anchor_domains: dict[int, str] = {}
    domain_counts: Counter[str] = Counter()
    scores: list[float] = []
    seen: set[tuple[str, int]] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object at {manifest_path}:{row_number}"
                )
            video_path = str(row.get("path", ""))
            if video_path not in positives:
                raise ValueError(
                    "Hard-positive manifest contains a non-positive video: "
                    f"{video_path!r}"
                )
            anchor_window = int(row["anchor_window_index"])
            key = (video_path, anchor_window)
            if key in seen:
                raise ValueError(f"Duplicate hard-positive anchor: {key}")
            if key not in key_to_index:
                raise ValueError(
                    "Hard-positive anchor is absent from selected training "
                    f"data: {key}"
                )
            seen.add(key)
            anchor_index = key_to_index[key]
            domain = str(row.get("domain", "unknown"))
            anchors[video_path].append(anchor_index)
            anchor_domains[anchor_index] = domain
            domain_counts[domain] += 1
            if "video_score" in row:
                scores.append(float(row["video_score"]))

            chunk_windows = row.get("window_indices")
            if not isinstance(chunk_windows, list) or not chunk_windows:
                raise ValueError(
                    f"window_indices must be non-empty at row {row_number}"
                )
            for value in chunk_windows:
                chunk_key = (video_path, int(value))
                if chunk_key not in key_to_index:
                    raise ValueError(
                        "Hard-positive chunk window is absent from selected "
                        f"training data: {chunk_key}"
                    )

    if not seen:
        raise ValueError(f"Hard-positive manifest is empty: {manifest_path}")
    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "candidate_count": len(seen),
        "video_count": len(anchors),
        "domain_counts": dict(sorted(domain_counts.items())),
    }
    if scores:
        summary["video_score_minimum"] = min(scores)
        summary["video_score_mean"] = sum(scores) / len(scores)
        summary["video_score_maximum"] = max(scores)
    return HardPositiveRegistry(
        anchors={path: tuple(values) for path, values in anchors.items()},
        anchor_domains=anchor_domains,
        summary=summary,
    )
