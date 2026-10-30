"""Batch samplers for temporal and identity-aware training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterator, Sequence

from torch.utils.data import Sampler

from edgefall_st.data.labels import FALL_ACTION_ID, FALLEN_ACTION_ID


@dataclass(frozen=True)
class _HardChunk:
    path: str
    indices: tuple[int, ...]
    category: str


class VideoGroupedBatchSampler(Sampler[list[int]]):
    """Build batches from adjacent windows of several distinct videos.

    Each video contributes ``windows_per_video`` temporally adjacent samples.
    Combining several videos in one batch provides both same-identity positives
    and different-video negatives. Optional event balancing reserves fixed
    video slots for chunks centered on an observable fall window.
    """

    def __init__(
        self,
        windows: Sequence[dict[str, Any]],
        *,
        batch_size: int,
        windows_per_video: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        positive_paths: Collection[str] | None = None,
        weak_positive_paths: Collection[str] | None = None,
        positive_videos_per_batch: int = 0,
        event_stage_balanced: bool = False,
        semantic_hard_anchors: dict[str, Sequence[int]] | None = None,
        score_hard_anchors: dict[str, Sequence[int]] | None = None,
        hard_anchor_categories: dict[int, str] | None = None,
        hard_positive_anchors: dict[str, Sequence[int]] | None = None,
        hard_positive_anchor_domains: dict[int, str] | None = None,
        hard_positive_videos_per_batch: int = 0,
        semantic_hard_videos_per_batch: int = 0,
        score_hard_videos_per_batch: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if windows_per_video <= 0:
            raise ValueError("windows_per_video must be positive")
        if batch_size % windows_per_video != 0:
            raise ValueError(
                "batch_size must be divisible by windows_per_video"
            )
        if not windows:
            raise ValueError("windows cannot be empty")

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, window in enumerate(windows):
            if "path" not in window or "end_time_s" not in window:
                raise ValueError(
                    "Every window needs path and end_time_s fields"
                )
            grouped[str(window["path"])].append(index)
        for indices in grouped.values():
            indices.sort(key=lambda index: float(windows[index]["end_time_s"]))

        self.windows = windows
        self.batch_size = int(batch_size)
        self.windows_per_video = int(windows_per_video)
        self.videos_per_batch = batch_size // windows_per_video
        if not 0 <= positive_videos_per_batch < self.videos_per_batch:
            raise ValueError(
                "positive_videos_per_batch must be in "
                "[0, videos_per_batch)"
            )
        if positive_videos_per_batch and not positive_paths:
            raise ValueError(
                "positive_paths are required for event-balanced sampling"
            )
        if positive_videos_per_batch and not drop_last:
            raise ValueError(
                "event-balanced sampling requires drop_last=True"
            )
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self._grouped_indices = dict(grouped)
        self.positive_paths = {
            str(path) for path in (positive_paths or ())
        }
        self.weak_positive_paths = {
            str(path) for path in (weak_positive_paths or ())
        }
        if not self.weak_positive_paths.issubset(self.positive_paths):
            raise ValueError("weak_positive_paths must be positive paths")
        self.positive_videos_per_batch = int(positive_videos_per_batch)
        self.event_stage_balanced = bool(event_stage_balanced)
        self.semantic_hard_anchors = {
            str(path): tuple(int(value) for value in values)
            for path, values in (semantic_hard_anchors or {}).items()
        }
        self.score_hard_anchors = {
            str(path): tuple(int(value) for value in values)
            for path, values in (score_hard_anchors or {}).items()
        }
        self.hard_anchor_categories = {
            int(index): str(category)
            for index, category in (hard_anchor_categories or {}).items()
        }
        self.hard_positive_anchors = {
            str(path): tuple(int(value) for value in values)
            for path, values in (hard_positive_anchors or {}).items()
        }
        self.hard_positive_anchor_domains = {
            int(index): str(domain)
            for index, domain in (hard_positive_anchor_domains or {}).items()
        }
        self.hard_positive_videos_per_batch = int(
            hard_positive_videos_per_batch
        )
        self.semantic_hard_videos_per_batch = int(
            semantic_hard_videos_per_batch
        )
        self.score_hard_videos_per_batch = int(
            score_hard_videos_per_batch
        )
        hard_slots = (
            self.semantic_hard_videos_per_batch
            + self.score_hard_videos_per_batch
        )
        negative_slots = self.videos_per_batch - self.positive_videos_per_batch
        if min(
            self.hard_positive_videos_per_batch,
            self.semantic_hard_videos_per_batch,
            self.score_hard_videos_per_batch,
        ) < 0:
            raise ValueError("Hard-negative video slots cannot be negative")
        if self.hard_positive_videos_per_batch > self.positive_videos_per_batch:
            raise ValueError(
                "Hard-positive video slots exceed positive batch capacity"
            )
        if self.hard_positive_videos_per_batch and not self.hard_positive_anchors:
            raise ValueError("Hard-positive anchors are required")
        if hard_slots > negative_slots:
            raise ValueError(
                "Hard-negative video slots exceed negative batch capacity"
            )
        if hard_slots and not self.positive_videos_per_batch:
            raise ValueError(
                "Hard-negative quotas require event-balanced sampling"
            )
        if self.semantic_hard_videos_per_batch and not self.semantic_hard_anchors:
            raise ValueError("Semantic hard-negative anchors are required")
        if self.score_hard_videos_per_batch and not self.score_hard_anchors:
            raise ValueError("Score hard-negative anchors are required")
        for path, anchor_indices in (
            *self.semantic_hard_anchors.items(),
            *self.score_hard_anchors.items(),
        ):
            if path in self.positive_paths:
                raise ValueError(
                    f"Hard-negative path is labeled positive: {path!r}"
                )
            grouped_indices = self._grouped_indices.get(path)
            if grouped_indices is None:
                raise ValueError(
                    f"Hard-negative path is absent from windows: {path!r}"
                )
            available = set(grouped_indices)
            if not anchor_indices or not set(anchor_indices).issubset(available):
                raise ValueError(
                    f"Invalid hard-negative anchors for path {path!r}"
                )
        for path, anchor_indices in self.hard_positive_anchors.items():
            if path not in self.positive_paths:
                raise ValueError(
                    f"Hard-positive path is not labeled positive: {path!r}"
                )
            grouped_indices = self._grouped_indices.get(path)
            if grouped_indices is None:
                raise ValueError(
                    f"Hard-positive path is absent from windows: {path!r}"
                )
            if not anchor_indices or not set(anchor_indices).issubset(
                set(grouped_indices)
            ):
                raise ValueError(
                    f"Invalid hard-positive anchors for path {path!r}"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _video_chunks(
        self,
        indices: list[int],
        *,
        start_offset: int = 0,
    ) -> list[list[int]]:
        if start_offset < 0:
            raise ValueError("start_offset cannot be negative")
        if self.drop_last and indices and len(indices) < self.windows_per_video:
            return [
                indices
                + [indices[-1]] * (self.windows_per_video - len(indices))
            ]
        selected = indices
        if self.drop_last:
            usable_count = (
                len(indices) // self.windows_per_video
            ) * self.windows_per_video
            if start_offset > len(indices) - usable_count:
                raise ValueError("start_offset leaves too few complete windows")
            selected = indices[start_offset : start_offset + usable_count]
        chunks = [
            selected[start : start + self.windows_per_video]
            for start in range(0, len(selected), self.windows_per_video)
        ]
        return chunks

    def _event_chunk(
        self,
        indices: list[int],
        random_generator: random.Random,
    ) -> list[int] | None:
        event_positions = [
            position
            for position, index in enumerate(indices)
            if int(self.windows[index].get("action_label", -1))
            == FALL_ACTION_ID
        ]
        if not event_positions or len(indices) < self.windows_per_video:
            return None
        anchor = (
            random_generator.choice(event_positions)
            if self.shuffle
            else event_positions[len(event_positions) // 2]
        )
        start = min(
            max(anchor - self.windows_per_video // 2, 0),
            len(indices) - self.windows_per_video,
        )
        return indices[start : start + self.windows_per_video]

    def _anchored_chunk(
        self,
        indices: list[int],
        anchor_index: int,
    ) -> list[int]:
        if len(indices) < self.windows_per_video:
            return indices + [indices[-1]] * (
                self.windows_per_video - len(indices)
            )
        try:
            anchor = indices.index(anchor_index)
        except ValueError as error:
            raise ValueError(
                f"Anchor index {anchor_index} is absent from its video"
            ) from error
        start = min(
            max(anchor - self.windows_per_video // 2, 0),
            len(indices) - self.windows_per_video,
        )
        return indices[start : start + self.windows_per_video]

    def _hard_chunks(
        self,
        anchors: dict[str, tuple[int, ...]],
        *,
        default_category: str,
    ) -> list[_HardChunk]:
        chunks: list[_HardChunk] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for path, anchor_indices in anchors.items():
            indices = self._grouped_indices[path]
            for anchor_index in anchor_indices:
                chunk = tuple(self._anchored_chunk(indices, anchor_index))
                key = (path, chunk)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(
                    _HardChunk(
                        path=path,
                        indices=chunk,
                        category=self.hard_anchor_categories.get(
                            anchor_index, default_category
                        ),
                    )
                )
        return chunks

    @staticmethod
    def _hard_drawer(
        chunks: Sequence[_HardChunk],
        random_generator: random.Random,
        *,
        sqrt_category_balance: bool,
    ):
        by_category: dict[str, list[_HardChunk]] = defaultdict(list)
        for chunk in chunks:
            by_category[chunk.category].append(chunk)
        for values in by_category.values():
            random_generator.shuffle(values)
        categories = sorted(by_category)
        weights = [
            (
                sqrt(len({chunk.path for chunk in by_category[category]}))
                if sqrt_category_balance
                else float(len(by_category[category]))
            )
            for category in categories
        ]
        cursors: defaultdict[str, int] = defaultdict(int)

        def draw(used_paths: set[str]) -> _HardChunk:
            maximum_attempts = max(32, len(chunks) * 2)
            for _ in range(maximum_attempts):
                category = random_generator.choices(
                    categories, weights=weights, k=1
                )[0]
                values = by_category[category]
                cursor = cursors[category]
                if cursor and cursor % len(values) == 0:
                    random_generator.shuffle(values)
                chunk = values[cursor % len(values)]
                cursors[category] = cursor + 1
                if chunk.path not in used_paths:
                    return chunk
            raise RuntimeError(
                "Could not draw a distinct hard-negative video for a batch"
            )

        return draw

    def _inject_hard_negative_chunks(
        self,
        batches: list[list[int]],
        random_generator: random.Random,
    ) -> list[list[int]]:
        semantic_slots = self.semantic_hard_videos_per_batch
        score_slots = self.score_hard_videos_per_batch
        if not semantic_slots and not score_slots:
            return batches
        semantic_chunks = self._hard_chunks(
            self.semantic_hard_anchors,
            default_category="semantic_hard_negative",
        )
        score_chunks = self._hard_chunks(
            self.score_hard_anchors,
            default_category="score_mined_unknown",
        )
        semantic_draw = self._hard_drawer(
            semantic_chunks,
            random_generator,
            sqrt_category_balance=True,
        )
        score_draw = self._hard_drawer(
            score_chunks,
            random_generator,
            sqrt_category_balance=False,
        )
        positive_windows = (
            self.positive_videos_per_batch * self.windows_per_video
        )
        negative_slots = (
            self.videos_per_batch - self.positive_videos_per_batch
        )
        ordinary_slots = negative_slots - semantic_slots - score_slots
        result: list[list[int]] = []
        for batch in batches:
            positive = batch[:positive_windows]
            negative = [
                batch[
                    positive_windows + slot * self.windows_per_video :
                    positive_windows + (slot + 1) * self.windows_per_video
                ]
                for slot in range(negative_slots)
            ]
            random_generator.shuffle(negative)
            selected = negative[:ordinary_slots]
            used_paths = {
                str(self.windows[index]["path"])
                for chunk in selected
                for index in chunk[:1]
            }
            used_paths.update(
                str(self.windows[index]["path"])
                for index in positive[:: self.windows_per_video]
            )
            for _ in range(semantic_slots):
                chunk = semantic_draw(used_paths)
                selected.append(list(chunk.indices))
                used_paths.add(chunk.path)
            for _ in range(score_slots):
                chunk = score_draw(used_paths)
                selected.append(list(chunk.indices))
                used_paths.add(chunk.path)
            rebuilt = positive + [index for chunk in selected for index in chunk]
            if len(rebuilt) != self.batch_size:
                raise RuntimeError(
                    "Hard-negative injection produced an incomplete batch"
                )
            result.append(rebuilt)
        return result

    def _inject_hard_positive_chunks(
        self,
        batches: list[list[int]],
        random_generator: random.Random,
    ) -> list[list[int]]:
        hard_slots = self.hard_positive_videos_per_batch
        if not hard_slots:
            return batches
        chunks = self._hard_chunks(
            self.hard_positive_anchors,
            default_category="hard_positive",
        )
        chunks = [
            _HardChunk(
                path=chunk.path,
                indices=chunk.indices,
                category=self.hard_positive_anchor_domains.get(
                    next(
                        anchor
                        for anchor in self.hard_positive_anchors[chunk.path]
                        if anchor in chunk.indices
                    ),
                    chunk.category,
                ),
            )
            for chunk in chunks
        ]
        draw = self._hard_drawer(
            chunks,
            random_generator,
            sqrt_category_balance=True,
        )
        positive_slots = self.positive_videos_per_batch
        positive_windows = positive_slots * self.windows_per_video
        ordinary_slots = positive_slots - hard_slots
        result: list[list[int]] = []
        for batch in batches:
            positive = [
                batch[slot * self.windows_per_video : (slot + 1) * self.windows_per_video]
                for slot in range(positive_slots)
            ]
            random_generator.shuffle(positive)
            selected = positive[:ordinary_slots]
            used_paths = {
                str(self.windows[chunk[0]]["path"]) for chunk in selected
            }
            for _ in range(hard_slots):
                chunk = draw(used_paths)
                selected.append(list(chunk.indices))
                used_paths.add(chunk.path)
            rebuilt = [index for chunk in selected for index in chunk]
            rebuilt.extend(batch[positive_windows:])
            if len(rebuilt) != self.batch_size:
                raise RuntimeError(
                    "Hard-positive injection produced an incomplete batch"
                )
            result.append(rebuilt)
        return result

    def _event_chunks(
        self,
        indices: list[int],
        random_generator: random.Random,
        *,
        weak_positive: bool = False,
    ) -> list[list[int]]:
        if weak_positive:
            chunks = self._video_chunks(indices)
            if self.shuffle:
                random_generator.shuffle(chunks)
            return chunks
        if not self.event_stage_balanced:
            chunk = self._event_chunk(indices, random_generator)
            return [] if chunk is None else [chunk]
        if len(indices) < self.windows_per_video:
            return []

        chunks_by_stage: dict[int, list[list[int]]] = {}
        for action_id in (FALL_ACTION_ID, FALLEN_ACTION_ID):
            candidates: list[list[int]] = []
            seen: set[tuple[int, ...]] = set()
            for anchor in (
                position
                for position, index in enumerate(indices)
                if int(self.windows[index].get("action_label", -1))
                == action_id
            ):
                start = min(
                    max(anchor - self.windows_per_video // 2, 0),
                    len(indices) - self.windows_per_video,
                )
                chunk = indices[start : start + self.windows_per_video]
                key = tuple(chunk)
                if key not in seen:
                    seen.add(key)
                    candidates.append(chunk)
            if self.shuffle:
                random_generator.shuffle(candidates)
            if candidates:
                chunks_by_stage[action_id] = candidates

        interleaved: list[list[int]] = []
        maximum_count = max(
            (len(chunks) for chunks in chunks_by_stage.values()),
            default=0,
        )
        for index in range(maximum_count):
            for action_id in (FALL_ACTION_ID, FALLEN_ACTION_ID):
                chunks = chunks_by_stage.get(action_id, ())
                if index < len(chunks):
                    interleaved.append(chunks[index])
        return interleaved

    def _expanded_paths(
        self,
        paths: list[str],
        required_count: int,
        video_chunks: dict[str, list[list[int]]],
        random_generator: random.Random,
    ) -> list[str]:
        if not paths:
            raise ValueError("Cannot expand an empty video path pool")
        expanded: list[str] = []
        while len(expanded) < required_count:
            cycle = list(paths)
            if self.shuffle:
                random_generator.shuffle(cycle)
            cycle.sort(
                key=lambda path: len(video_chunks[path]),
                reverse=True,
            )
            expanded.extend(cycle)
        return expanded[:required_count]

    def _build_event_balanced_batches(
        self,
        video_chunks: dict[str, list[list[int]]],
        random_generator: random.Random,
    ) -> list[list[int]]:
        positive_slots = self.positive_videos_per_batch
        negative_slots = self.videos_per_batch - positive_slots
        event_chunks = {
            path: chunks
            for path, indices in self._grouped_indices.items()
            if path in self.positive_paths
            and (
                chunks := self._event_chunks(
                    indices,
                    random_generator,
                    weak_positive=path in self.weak_positive_paths,
                )
            )
        }
        positive_paths = list(event_chunks)
        negative_paths = [
            path
            for path in video_chunks
            if path not in self.positive_paths
        ]
        if len(positive_paths) < positive_slots:
            raise ValueError(
                "Not enough event videos with a sampled fall window for "
                "event-balanced sampling"
            )
        if len(negative_paths) < negative_slots:
            raise ValueError(
                "Not enough negative videos for event-balanced sampling"
            )

        cohort_count = max(
            (len(positive_paths) + positive_slots - 1) // positive_slots,
            (len(negative_paths) + negative_slots - 1) // negative_slots,
        )
        expanded_positive = self._expanded_paths(
            positive_paths,
            cohort_count * positive_slots,
            event_chunks,
            random_generator,
        )
        expanded_negative = self._expanded_paths(
            negative_paths,
            cohort_count * negative_slots,
            video_chunks,
            random_generator,
        )

        batches: list[list[int]] = []
        event_chunk_cursors: defaultdict[str, int] = defaultdict(int)
        for cohort_index in range(cohort_count):
            positive_cohort = expanded_positive[
                cohort_index * positive_slots :
                (cohort_index + 1) * positive_slots
            ]
            negative_cohort = expanded_negative[
                cohort_index * negative_slots :
                (cohort_index + 1) * negative_slots
            ]
            round_count = min(
                len(video_chunks[path]) for path in negative_cohort
            )
            for round_index in range(round_count):
                batch: list[int] = []
                for path in positive_cohort:
                    chunks = event_chunks[path]
                    cursor = event_chunk_cursors[path]
                    batch.extend(chunks[cursor % len(chunks)])
                    event_chunk_cursors[path] = cursor + 1
                batch.extend(
                    index
                    for path in negative_cohort
                    for index in video_chunks[path][round_index]
                )
                if len(batch) != self.batch_size:
                    raise RuntimeError(
                        "Event-balanced sampler produced an incomplete batch"
                    )
                batches.append(batch)
        return batches

    def _build_batches(self, *, epoch: int) -> list[list[int]]:
        random_generator = random.Random(self.seed + epoch)
        video_chunks: dict[str, list[list[int]]] = {}
        for path, indices in self._grouped_indices.items():
            remainder = len(indices) % self.windows_per_video
            start_offset = (
                random_generator.randint(0, remainder)
                if self.shuffle and self.drop_last and remainder
                else 0
            )
            chunks = self._video_chunks(
                indices,
                start_offset=start_offset,
            )
            if self.shuffle:
                random_generator.shuffle(chunks)
            if chunks:
                video_chunks[path] = chunks

        if self.positive_videos_per_batch:
            batches = self._build_event_balanced_batches(
                video_chunks,
                random_generator,
            )
            batches = self._inject_hard_positive_chunks(
                batches, random_generator
            )
            return self._inject_hard_negative_chunks(
                batches, random_generator
            )

        active_paths = list(video_chunks)
        if self.shuffle:
            random_generator.shuffle(active_paths)
            # Stable sorting keeps the randomized order among videos with the
            # same number of chunks while minimizing cohort tail drops.
            active_paths.sort(
                key=lambda path: len(video_chunks[path]),
                reverse=True,
            )
        batches: list[list[int]] = []
        for start in range(0, len(active_paths), self.videos_per_batch):
            cohort = active_paths[start : start + self.videos_per_batch]
            if self.drop_last and len(cohort) < self.videos_per_batch:
                break
            round_count = (
                min(len(video_chunks[path]) for path in cohort)
                if self.drop_last
                else max(len(video_chunks[path]) for path in cohort)
            )
            for round_index in range(round_count):
                batch = [
                    index
                    for path in cohort
                    if round_index < len(video_chunks[path])
                    for index in video_chunks[path][round_index]
                ]
                if len(batch) == self.batch_size or (
                    not self.drop_last and batch
                ):
                    batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches(epoch=self.epoch)

    def __len__(self) -> int:
        return len(self._build_batches(epoch=self.epoch))
