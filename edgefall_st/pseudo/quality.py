"""Per-video and aggregate pseudo-track quality statistics."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VideoTrackQuality:
    path: str
    expected_frames: int
    processed_frames: int
    fps: float
    width: int
    height: int
    frames_with_person: int
    multi_detection_frames: int
    untracked_detection_frames: int
    unique_track_ids: int
    dominant_track_frames: int
    mean_person_confidence: float
    median_bbox_area_ratio: float
    elapsed_s: float

    @property
    def detection_coverage(self) -> float:
        return self.frames_with_person / max(self.processed_frames, 1)

    @property
    def dominant_track_coverage(self) -> float:
        return self.dominant_track_frames / max(self.processed_frames, 1)

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["detection_coverage"] = self.detection_coverage
        values["dominant_track_coverage"] = self.dominant_track_coverage
        return values


def write_quality(quality: VideoTrackQuality, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(quality.as_dict(), handle, ensure_ascii=False, indent=2)
    temporary.replace(output)


def load_quality(path: str | Path) -> VideoTrackQuality:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if (
        "multi_detection_frames" not in values
        and "multi_person_frames" in values
    ):
        values["multi_detection_frames"] = values["multi_person_frames"]
    fields = {
        name: values[name]
        for name in VideoTrackQuality.__dataclass_fields__
    }
    return VideoTrackQuality(**fields)


def summarize_quality(
    qualities: Iterable[VideoTrackQuality],
) -> dict[str, object]:
    rows = list(qualities)
    if not rows:
        return {"processed_videos": 0}
    detection_coverages = [row.detection_coverage for row in rows]
    dominant_coverages = [row.dominant_track_coverage for row in rows]
    ordered_detection = sorted(detection_coverages)
    ordered_dominant = sorted(dominant_coverages)
    p10_index = max(0, round((len(ordered_detection) - 1) * 0.10))
    grouped_rows: defaultdict[str, list[VideoTrackQuality]] = defaultdict(list)
    for row in rows:
        grouped_rows[Path(row.path).parts[0]].append(row)
    source_action_summary = {}
    for source_action, group in sorted(grouped_rows.items()):
        source_action_summary[source_action] = {
            "videos": len(group),
            "mean_detection_coverage": statistics.fmean(
                row.detection_coverage for row in group
            ),
            "mean_dominant_track_coverage": statistics.fmean(
                row.dominant_track_coverage for row in group
            ),
            "videos_below_95pct_detection": sum(
                row.detection_coverage < 0.95 for row in group
            ),
            "videos_below_90pct_dominant": sum(
                row.dominant_track_coverage < 0.90 for row in group
            ),
        }
    return {
        "processed_videos": len(rows),
        "processed_frames": sum(row.processed_frames for row in rows),
        "mean_detection_coverage": statistics.fmean(detection_coverages),
        "median_detection_coverage": statistics.median(detection_coverages),
        "p10_detection_coverage": ordered_detection[p10_index],
        "mean_dominant_track_coverage": statistics.fmean(dominant_coverages),
        "median_dominant_track_coverage": statistics.median(
            dominant_coverages
        ),
        "p10_dominant_track_coverage": ordered_dominant[p10_index],
        "videos_below_95pct_detection": [
            row.path for row in rows if row.detection_coverage < 0.95
        ],
        "videos_below_90pct_dominant": [
            row.path for row in rows if row.dominant_track_coverage < 0.90
        ],
        "videos_below_95pct_dominant": [
            row.path for row in rows if row.dominant_track_coverage < 0.95
        ],
        "videos_with_multiple_track_ids": [
            row.path for row in rows if row.unique_track_ids > 1
        ],
        "videos_with_multi_detection_frames": [
            row.path for row in rows if row.multi_detection_frames > 0
        ],
        "mean_person_confidence": statistics.fmean(
            row.mean_person_confidence for row in rows
        ),
        "mean_multi_detection_frame_ratio": statistics.fmean(
            row.multi_detection_frames / max(row.processed_frames, 1)
            for row in rows
        ),
        "mean_elapsed_s_per_video": statistics.fmean(
            row.elapsed_s for row in rows
        ),
        "source_action_summary": source_action_summary,
    }
