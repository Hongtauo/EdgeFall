"""Deterministic, coverage-aware pilot video sampling."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

METADATA_KEYS = (
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
class PilotVideo:
    path: str
    split: str
    labels: tuple[int, ...]
    metadata: dict[str, str]

    @property
    def folder(self) -> str:
        return self.path.split("/", maxsplit=1)[0]

    def tokens(self) -> tuple[str, ...]:
        values = {f"label:{label}" for label in self.labels}
        values.add(f"folder:{self.folder}")
        for key in METADATA_KEYS:
            value = self.metadata.get(key)
            if value:
                values.add(f"{key}:{value}")
        return tuple(sorted(values))


def load_split_manifest(
    manifest_path: str | Path,
    *,
    split: str = "train",
) -> list[PilotVideo]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "split", "labels", *METADATA_KEYS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Split manifest is missing fields: {sorted(missing)}")
        videos = []
        for row in reader:
            if row["split"] != split:
                continue
            labels = tuple(
                sorted(int(value) for value in row["labels"].split("|") if value)
            )
            videos.append(
                PilotVideo(
                    path=row["path"].replace("\\", "/"),
                    split=row["split"],
                    labels=labels,
                    metadata={key: row.get(key, "") for key in METADATA_KEYS},
                )
            )
    if not videos:
        raise ValueError(f"No videos found for split={split!r} in {path}")
    return videos


def _token_targets(
    videos: Sequence[PilotVideo],
    sample_size: int,
    *,
    minimum_per_label: int,
) -> dict[str, int]:
    token_totals = Counter(token for video in videos for token in video.tokens())
    targets: dict[str, int] = {}
    folder_count = len({video.folder for video in videos})
    age_count = len(
        {
            video.metadata["age_group"]
            for video in videos
            if video.metadata.get("age_group")
        }
    )
    for token, total in token_totals.items():
        if token.startswith("label:"):
            targets[token] = min(total, minimum_per_label)
        elif token.startswith("folder:"):
            targets[token] = min(total, math.ceil(sample_size / folder_count))
        elif token.startswith("age_group:"):
            targets[token] = min(total, math.ceil(sample_size / age_count))
        else:
            proportional = max(1, round(sample_size * total / len(videos)))
            targets[token] = min(total, proportional)
    return targets


def select_pilot_videos(
    videos: Sequence[PilotVideo],
    *,
    sample_size: int = 160,
    seed: int = 20260718,
    minimum_per_label: int = 10,
) -> list[PilotVideo]:
    """Greedily cover all actions and important metadata categories."""

    if not 0 < sample_size <= len(videos):
        raise ValueError("sample_size must be in [1, number of candidate videos]")
    if minimum_per_label <= 0:
        raise ValueError("minimum_per_label must be positive")
    if sample_size == len(videos):
        selected = sorted(videos, key=lambda video: video.path)
        random.Random(seed).shuffle(selected)
        return selected

    targets = _token_targets(
        videos,
        sample_size,
        minimum_per_label=minimum_per_label,
    )
    token_totals = Counter(token for video in videos for token in video.tokens())
    selected_counts: Counter[str] = Counter()
    selected: list[PilotVideo] = []
    remaining = {video.path: video for video in videos}
    rng = random.Random(seed)
    tie_breakers = {video.path: rng.random() for video in videos}

    def token_weight(token: str) -> float:
        if token.startswith("label:"):
            label = int(token.split(":", maxsplit=1)[1])
            return 5.0 if label >= 10 else 3.0
        if token.startswith(("folder:", "age_group:")):
            return 1.5
        return 0.5

    while len(selected) < sample_size:
        best_video: PilotVideo | None = None
        best_score: tuple[float, float, str] | None = None
        for video in remaining.values():
            coverage_score = 0.0
            rarity_score = 0.0
            for token in video.tokens():
                target = targets[token]
                deficit = max(0, target - selected_counts[token])
                coverage_score += token_weight(token) * deficit / target
                rarity_score += 1.0 / token_totals[token]
            score = (
                coverage_score + 0.05 * rarity_score,
                tie_breakers[video.path],
                video.path,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_video = video
        if best_video is None:
            raise RuntimeError("Pilot selection exhausted candidates unexpectedly")
        selected.append(best_video)
        selected_counts.update(best_video.tokens())
        del remaining[best_video.path]

    label_counts = Counter(
        label for video in selected for label in set(video.labels)
    )
    available_labels = Counter(
        label for video in videos for label in set(video.labels)
    )
    missing_targets = {
        label: min(minimum_per_label, available)
        for label, available in available_labels.items()
        if label_counts[label] < min(minimum_per_label, available)
    }
    if missing_targets:
        raise RuntimeError(
            "Pilot selection did not meet label coverage targets: "
            f"{missing_targets}"
        )
    return selected


def write_pilot_manifest(
    videos: Iterable[PilotVideo],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["order", "path", "split", "labels", "folder", *METADATA_KEYS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for order, video in enumerate(videos):
            row: dict[str, object] = {
                "order": order,
                "path": video.path,
                "split": video.split,
                "labels": "|".join(str(label) for label in video.labels),
                "folder": video.folder,
            }
            row.update(video.metadata)
            writer.writerow(row)


def read_pilot_manifest(path: str | Path) -> list[PilotVideo]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["order"]))
        return [
            PilotVideo(
                path=row["path"].replace("\\", "/"),
                split=row["split"],
                labels=tuple(
                    sorted(
                        int(value)
                        for value in row["labels"].split("|")
                        if value
                    )
                ),
                metadata={key: row.get(key, "") for key in METADATA_KEYS},
            )
            for row in rows
        ]
