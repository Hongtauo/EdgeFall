"""Load and validate chunk-level hard-negative training manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence


HARD_NEGATIVE_SEMANTIC = "semantic"
HARD_NEGATIVE_SCORE = "score"
HARD_NEGATIVE_KINDS = (
    HARD_NEGATIVE_SEMANTIC,
    HARD_NEGATIVE_SCORE,
)


@dataclass(frozen=True)
class HardNegativeRegistry:
    """Resolved hard-negative anchors for one combined training dataset."""

    semantic_anchors: Mapping[str, tuple[int, ...]]
    score_anchors: Mapping[str, tuple[int, ...]]
    anchor_categories: Mapping[int, str]
    anchor_window_categories: Mapping[tuple[str, int], str]
    window_categories: Mapping[tuple[str, int], tuple[str, ...]]
    window_keys: frozenset[tuple[str, int]]
    summary: Mapping[str, Any]

    @property
    def enabled(self) -> bool:
        return bool(self.semantic_anchors or self.score_anchors)


def empty_hard_negative_registry() -> HardNegativeRegistry:
    return HardNegativeRegistry(
        semantic_anchors={},
        score_anchors={},
        anchor_categories={},
        anchor_window_categories={},
        window_categories={},
        window_keys=frozenset(),
        summary={
            "candidate_count": 0,
            "semantic_candidate_count": 0,
            "score_candidate_count": 0,
            "hard_window_count": 0,
        },
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected an object at {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"Hard-negative manifest is empty: {path}")
    return rows


def load_hard_negative_registry(
    path: str | Path,
    windows: Sequence[dict[str, Any]],
    *,
    positive_paths: Collection[str],
) -> HardNegativeRegistry:
    """Resolve manifest window IDs to indices in the selected train caches."""

    manifest_path = Path(path)
    key_to_index: dict[tuple[str, int], int] = {}
    for index, window in enumerate(windows):
        if "window_index" not in window:
            raise ValueError(
                "Training windows must expose window_index for hard negatives"
            )
        key = (str(window["path"]), int(window["window_index"]))
        if key in key_to_index:
            raise ValueError(f"Duplicate selected training window: {key}")
        key_to_index[key] = index

    positives = {str(value) for value in positive_paths}
    anchors: dict[str, dict[str, list[int]]] = {
        kind: defaultdict(list) for kind in HARD_NEGATIVE_KINDS
    }
    anchor_categories: dict[int, str] = {}
    anchor_window_categories: dict[tuple[str, int], str] = {}
    hard_window_categories: dict[tuple[str, int], set[str]] = defaultdict(set)
    hard_window_keys: set[tuple[str, int]] = set()
    category_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    seen_anchors: set[tuple[str, str, int]] = set()

    for row_number, row in enumerate(_read_rows(manifest_path), start=1):
        kind = str(row.get("kind", ""))
        if kind not in HARD_NEGATIVE_KINDS:
            raise ValueError(
                f"Invalid hard-negative kind at row {row_number}: {kind!r}"
            )
        video_path = str(row.get("path", ""))
        if not video_path:
            raise ValueError(f"Missing path at row {row_number}")
        if video_path in positives:
            raise ValueError(
                f"Hard-negative manifest contains positive video {video_path!r}"
            )
        anchor_window = int(row["anchor_window_index"])
        anchor_key = (video_path, anchor_window)
        if anchor_key not in key_to_index:
            raise ValueError(
                "Hard-negative anchor is absent from selected training data: "
                f"{anchor_key}"
            )
        candidate_key = (kind, video_path, anchor_window)
        if candidate_key in seen_anchors:
            raise ValueError(f"Duplicate hard-negative anchor: {candidate_key}")
        seen_anchors.add(candidate_key)

        category = str(row.get("category", "score_mined_unknown"))
        anchor_index = key_to_index[anchor_key]
        anchors[kind][video_path].append(anchor_index)
        anchor_categories[anchor_index] = category
        if kind == HARD_NEGATIVE_SEMANTIC:
            previous_category = anchor_window_categories.get(anchor_key)
            if previous_category is not None and previous_category != category:
                raise ValueError(
                    "Hard-negative anchor has conflicting semantic categories: "
                    f"{anchor_key}: {previous_category!r} != {category!r}"
                )
            anchor_window_categories[anchor_key] = category
        category_counts[category] += 1
        domain_counts[str(row.get("domain", "unknown"))] += 1

        chunk_windows = row.get("window_indices")
        if not isinstance(chunk_windows, list) or not chunk_windows:
            raise ValueError(
                f"window_indices must be a non-empty list at row {row_number}"
            )
        for value in chunk_windows:
            key = (video_path, int(value))
            if key not in key_to_index:
                raise ValueError(
                    "Hard-negative chunk window is absent from selected "
                    f"training data: {key}"
                )
            hard_window_keys.add(key)
            hard_window_categories[key].add(category)

    resolved = {
        kind: {
            video_path: tuple(values)
            for video_path, values in grouped.items()
        }
        for kind, grouped in anchors.items()
    }
    summary = {
        "manifest": str(manifest_path),
        "candidate_count": len(seen_anchors),
        "semantic_candidate_count": sum(
            len(values) for values in resolved[HARD_NEGATIVE_SEMANTIC].values()
        ),
        "score_candidate_count": sum(
            len(values) for values in resolved[HARD_NEGATIVE_SCORE].values()
        ),
        "semantic_video_count": len(resolved[HARD_NEGATIVE_SEMANTIC]),
        "score_video_count": len(resolved[HARD_NEGATIVE_SCORE]),
        "hard_window_count": len(hard_window_keys),
        "category_counts": dict(sorted(category_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }
    return HardNegativeRegistry(
        semantic_anchors=resolved[HARD_NEGATIVE_SEMANTIC],
        score_anchors=resolved[HARD_NEGATIVE_SCORE],
        anchor_categories=anchor_categories,
        anchor_window_categories=anchor_window_categories,
        window_categories={
            key: tuple(sorted(values))
            for key, values in hard_window_categories.items()
        },
        window_keys=frozenset(hard_window_keys),
        summary=summary,
    )
