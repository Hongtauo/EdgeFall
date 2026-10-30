"""Load, validate, and clean OmniFall temporal annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from edgefall_st.data.labels import ACTION_LABELS

CORE_FIELDS = ("path", "label", "start", "end")


@dataclass(frozen=True)
class Segment:
    """A left-closed, right-open temporal action segment."""

    path: str
    label: int
    start: float
    end: float
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)
    source_row: int = field(default=-1, compare=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VideoAnnotations:
    path: str
    segments: tuple[Segment, ...]

    @property
    def duration(self) -> float:
        return max(segment.end for segment in self.segments)

    @property
    def labels(self) -> frozenset[int]:
        return frozenset(segment.label for segment in self.segments)

    @property
    def metadata(self) -> Mapping[str, str]:
        return self.segments[0].metadata


@dataclass
class CleaningReport:
    input_rows: int = 0
    output_rows: int = 0
    dropped_invalid: int = 0
    dropped_duplicates: int = 0
    merged_same_label: int = 0
    snapped_boundaries: int = 0
    large_gaps: int = 0
    large_overlaps: int = 0
    issues: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "dropped_invalid": self.dropped_invalid,
            "dropped_duplicates": self.dropped_duplicates,
            "merged_same_label": self.merged_same_label,
            "snapped_boundaries": self.snapped_boundaries,
            "large_gaps": self.large_gaps,
            "large_overlaps": self.large_overlaps,
            "issues": self.issues,
        }


class AmbiguousAnnotationError(ValueError):
    """Raised when multiple action labels cover the same timestamp."""


def load_annotations(csv_path: str | Path) -> tuple[list[Segment], list[str]]:
    """Load the OmniFall CSV while preserving metadata column order."""

    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Annotation file has no header: {path}")
        missing = [name for name in CORE_FIELDS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required annotation fields: {missing}")

        metadata_fields = [
            name for name in reader.fieldnames if name not in CORE_FIELDS
        ]
        segments: list[Segment] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                start = float(row["start"])
                end = float(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid annotation values at {path}:{row_number}"
                ) from exc
            if label not in ACTION_LABELS:
                raise ValueError(
                    f"Unknown action id {label} at {path}:{row_number}"
                )
            metadata = {name: row.get(name, "") for name in metadata_fields}
            segments.append(
                Segment(
                    path=row["path"].replace("\\", "/"),
                    label=label,
                    start=start,
                    end=end,
                    metadata=metadata,
                    source_row=row_number,
                )
            )
    return segments, metadata_fields


def clean_annotations(
    segments: Sequence[Segment],
    *,
    boundary_fps: float = 30.0,
    same_label_tolerance_s: float = 1e-6,
) -> tuple[list[VideoAnnotations], CleaningReport]:
    """Clean segments and align sub-frame boundary disagreements.

    Large gaps are preserved as unlabeled time. Large overlaps are reported and
    remain ambiguous so downstream labeling cannot silently choose a class.
    """

    if boundary_fps <= 0:
        raise ValueError("boundary_fps must be positive")
    frame_tolerance = 1.0 / boundary_fps
    report = CleaningReport(input_rows=len(segments))

    valid: list[Segment] = []
    seen: set[tuple[str, int, float, float]] = set()
    for segment in segments:
        if segment.start >= segment.end:
            report.dropped_invalid += 1
            report.issues.append(
                {
                    "kind": "invalid_duration",
                    "path": segment.path,
                    "label": segment.label,
                    "start": segment.start,
                    "end": segment.end,
                    "source_row": segment.source_row,
                }
            )
            continue
        key = (segment.path, segment.label, segment.start, segment.end)
        if key in seen:
            report.dropped_duplicates += 1
            report.issues.append(
                {
                    "kind": "exact_duplicate",
                    "path": segment.path,
                    "label": segment.label,
                    "start": segment.start,
                    "end": segment.end,
                    "source_row": segment.source_row,
                }
            )
            continue
        seen.add(key)
        valid.append(segment)

    grouped: dict[str, list[Segment]] = {}
    for segment in valid:
        grouped.setdefault(segment.path, []).append(segment)

    videos: list[VideoAnnotations] = []
    for video_path in sorted(grouped):
        ordered = sorted(
            grouped[video_path],
            key=lambda item: (item.start, item.end, item.label),
        )
        merged: list[Segment] = []
        for segment in ordered:
            if (
                merged
                and merged[-1].label == segment.label
                and segment.start <= merged[-1].end + same_label_tolerance_s
            ):
                previous = merged[-1]
                merged[-1] = replace(previous, end=max(previous.end, segment.end))
                report.merged_same_label += 1
            else:
                merged.append(segment)

        aligned = list(merged)
        for index in range(1, len(aligned)):
            previous = aligned[index - 1]
            current = aligned[index]
            delta = current.start - previous.end
            if delta == 0:
                continue
            if abs(delta) <= frame_tolerance:
                midpoint = (previous.end + current.start) / 2.0
                aligned[index - 1] = replace(previous, end=midpoint)
                aligned[index] = replace(current, start=midpoint)
                report.snapped_boundaries += 1
            elif delta > 0:
                report.large_gaps += 1
                report.issues.append(
                    {
                        "kind": "large_gap",
                        "path": video_path,
                        "previous_label": previous.label,
                        "next_label": current.label,
                        "previous_end": previous.end,
                        "next_start": current.start,
                        "delta_s": delta,
                    }
                )
            else:
                report.large_overlaps += 1
                report.issues.append(
                    {
                        "kind": "large_overlap",
                        "path": video_path,
                        "previous_label": previous.label,
                        "next_label": current.label,
                        "previous_end": previous.end,
                        "next_start": current.start,
                        "delta_s": delta,
                    }
                )

        videos.append(VideoAnnotations(video_path, tuple(aligned)))

    report.output_rows = sum(len(video.segments) for video in videos)
    return videos, report


def label_at(segments: Sequence[Segment], timestamp_s: float) -> int | None:
    """Return the action at a timestamp using [start, end) semantics."""

    matches = [
        segment for segment in segments if segment.start <= timestamp_s < segment.end
    ]
    if not matches:
        return None
    labels = {segment.label for segment in matches}
    if len(labels) > 1:
        raise AmbiguousAnnotationError(
            f"Conflicting labels {sorted(labels)} at t={timestamp_s:.6f}s "
            f"for {matches[0].path}"
        )
    return matches[0].label


def write_clean_annotations(
    videos: Iterable[VideoAnnotations],
    output_path: str | Path,
    metadata_fields: Sequence[str],
) -> None:
    """Write cleaned segments using the original OmniFall schema."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*CORE_FIELDS, *metadata_fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video in videos:
            for segment in video.segments:
                row: dict[str, object] = {
                    "path": segment.path,
                    "label": segment.label,
                    "start": f"{segment.start:.6f}",
                    "end": f"{segment.end:.6f}",
                }
                row.update({name: segment.metadata.get(name, "") for name in metadata_fields})
                writer.writerow(row)
