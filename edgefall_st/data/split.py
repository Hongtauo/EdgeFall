"""Deterministic video-level stratified splitting."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from edgefall_st.data.annotations import VideoAnnotations

DEFAULT_METADATA_KEYS = (
    "environment_category",
    "camera_shot",
    "speed",
    "camera_elevation",
    "camera_azimuth",
    "camera_distance",
    "age_group",
    "bmi_band",
)


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    val: float = 0.1
    test: float = 0.2

    def as_dict(self) -> dict[str, float]:
        values = {"train": self.train, "val": self.val, "test": self.test}
        if any(value <= 0 for value in values.values()):
            raise ValueError("All split ratios must be positive")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Split ratios must sum to 1")
        return values


def _target_sizes(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    targets = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(raw, key=lambda name: (raw[name] - targets[name], name), reverse=True)
    for name in order[:remainder]:
        targets[name] += 1
    return targets


def _tokens(
    video: VideoAnnotations,
    metadata_keys: Sequence[str],
) -> tuple[str, ...]:
    values = {f"label:{label}" for label in video.labels}
    for key in metadata_keys:
        value = video.metadata.get(key)
        if value:
            values.add(f"{key}:{value}")
    return tuple(sorted(values))


def stratified_video_split(
    videos: Sequence[VideoAnnotations],
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 20260717,
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
) -> dict[str, str]:
    """Assign each video to exactly one split with multi-label balancing."""

    ratio_map = ratios.as_dict()
    if not videos:
        return {}
    paths = [video.path for video in videos]
    if len(paths) != len(set(paths)):
        raise ValueError("Video paths must be unique before splitting")

    target_sizes = _target_sizes(len(videos), ratio_map)
    video_tokens = {video.path: _tokens(video, metadata_keys) for video in videos}
    token_totals = Counter(
        token for tokens in video_tokens.values() for token in tokens
    )
    token_targets = {
        split: {
            token: total * ratio_map[split] for token, total in token_totals.items()
        }
        for split in ratio_map
    }

    rng = random.Random(seed)
    random_tie = {video.path: rng.random() for video in videos}
    ordered = sorted(
        videos,
        key=lambda video: (
            -sum(1.0 / token_totals[token] for token in video_tokens[video.path]),
            random_tie[video.path],
            video.path,
        ),
    )

    split_counts = Counter()
    split_token_counts = {name: Counter() for name in ratio_map}
    assignments: dict[str, str] = {}

    for video in ordered:
        tokens = video_tokens[video.path]
        candidates = [
            name
            for name in ratio_map
            if split_counts[name] < target_sizes[name]
        ]
        if not candidates:
            raise RuntimeError("No split has remaining capacity")

        def score(split: str) -> tuple[float, float, str]:
            target_size = max(target_sizes[split], 1)
            size_need = (target_sizes[split] - split_counts[split]) / target_size
            token_needs = [
                (
                    token_targets[split][token]
                    - split_token_counts[split][token]
                )
                / max(token_targets[split][token], 1.0)
                for token in tokens
            ]
            token_need = sum(token_needs) / max(len(token_needs), 1)
            return (0.35 * size_need + 0.65 * token_need, rng.random(), split)

        selected = max(candidates, key=score)
        assignments[video.path] = selected
        split_counts[selected] += 1
        split_token_counts[selected].update(tokens)

    return assignments


def write_split_manifest(
    videos: Sequence[VideoAnnotations],
    assignments: Mapping[str, str],
    output_path: str | Path,
    *,
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "split", "labels", *metadata_keys]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video in sorted(videos, key=lambda item: item.path):
            row: dict[str, object] = {
                "path": video.path,
                "split": assignments[video.path],
                "labels": "|".join(str(label) for label in sorted(video.labels)),
            }
            row.update(
                {key: video.metadata.get(key, "") for key in metadata_keys}
            )
            writer.writerow(row)
