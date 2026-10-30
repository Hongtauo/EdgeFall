"""Dataset discovery and annotation parsing for CAUCAFall."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SOURCE_FPS = 20.0
ACTIVITY_LABELS = {
    "Fall backwards": True,
    "Fall forward": True,
    "Fall left": True,
    "Fall right": True,
    "Fall sitting": True,
    "Hop": False,
    "Kneel": False,
    "Pick up object": False,
    "Sit down": False,
    "Walk": False,
}
LIGHTING_BY_SUBJECT = {
    1: "natural_light",
    2: "natural_light",
    3: "natural_light",
    4: "natural_light",
    5: "artificial_light",
    6: "zero_lux",
    7: "artificial_light",
    8: "zero_lux",
    9: "artificial_light",
    10: "artificial_light",
}


class DatasetLayoutError(RuntimeError):
    """Raised when CAUCAFall does not match its documented layout."""


@dataclass(frozen=True)
class CaucaFrame:
    source_index: int
    image_path: Path
    annotation_path: Path | None
    pairing: str
    target: bool | None
    bbox_xyxy_norm: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class CaucaVideo:
    sample_id: str
    subject_id: int
    activity: str
    label: bool
    lighting: str
    video_path: Path
    frames: tuple[CaucaFrame, ...]

    @property
    def annotated_frames(self) -> int:
        return sum(frame.target is not None for frame in self.frames)

    @property
    def fall_frames(self) -> int:
        return sum(frame.target is True for frame in self.frames)

    @property
    def alias_pairs(self) -> int:
        return sum(frame.pairing == "numeric_alias" for frame in self.frames)

    @property
    def missing_annotations(self) -> int:
        return sum(frame.annotation_path is None for frame in self.frames)


def _subject_number(path: Path) -> int:
    match = re.fullmatch(r"Subject\.(\d+)", path.name, flags=re.IGNORECASE)
    if match is None:
        raise DatasetLayoutError(f"Unexpected subject directory: {path}")
    value = int(match.group(1))
    if value not in LIGHTING_BY_SUBJECT:
        raise DatasetLayoutError(f"Unsupported subject number: {value}")
    return value


def _numeric_identifier(stem: str) -> str | None:
    groups = re.findall(r"\d+", stem)
    return None if not groups else "".join(groups)


def _frame_sort_key(path: Path) -> tuple[int, str]:
    identifier = _numeric_identifier(path.stem)
    return (
        -1 if identifier is None else int(identifier),
        path.name.casefold(),
    )


def read_annotation(
    path: str | Path,
) -> tuple[bool, tuple[float, float, float, float]]:
    """Read one class-plus-person-box line in normalized YOLO format."""

    annotation_path = Path(path)
    lines = [
        line.strip()
        for line in annotation_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"Expected one annotation in {annotation_path}")
    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"Expected five annotation fields in {annotation_path}")
    try:
        class_id = int(fields[0])
        center_x, center_y, width, height = (float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"Invalid annotation values in {annotation_path}") from error
    if class_id not in (0, 1):
        raise ValueError(f"Unsupported class {class_id} in {annotation_path}")
    values = (center_x, center_y, width, height)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Annotation values must be normalized in {annotation_path}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Annotation box must have positive area in {annotation_path}")
    left = max(0.0, center_x - width * 0.5)
    top = max(0.0, center_y - height * 0.5)
    right = min(1.0, center_x + width * 0.5)
    bottom = min(1.0, center_y + height * 0.5)
    if right <= left or bottom <= top:
        raise ValueError(f"Clipped annotation box is empty in {annotation_path}")
    return bool(class_id), (left, top, right, bottom)


def _pair_frames(activity_root: Path) -> tuple[CaucaFrame, ...]:
    images = sorted(activity_root.glob("*.png"), key=_frame_sort_key)
    annotations = {
        path.stem.casefold(): path
        for path in activity_root.glob("*.txt")
        if path.name.casefold() != "classes.txt"
    }
    if not images:
        raise DatasetLayoutError(f"No PNG frames found in {activity_root}")

    used_annotations: set[Path] = set()
    frames: list[CaucaFrame] = []
    for source_index, image_path in enumerate(images):
        annotation_path = annotations.get(image_path.stem.casefold())
        pairing = "exact"
        if annotation_path is None:
            identifier = _numeric_identifier(image_path.stem)
            candidates = [
                path
                for path in annotations.values()
                if path not in used_annotations
                and identifier is not None
                and _numeric_identifier(path.stem) == identifier
            ]
            if len(candidates) == 1:
                annotation_path = candidates[0]
                pairing = "numeric_alias"
            elif len(candidates) > 1:
                raise DatasetLayoutError(
                    f"Ambiguous annotation aliases for {image_path}"
                )

        target: bool | None = None
        bbox: tuple[float, float, float, float] | None = None
        if annotation_path is not None:
            target, bbox = read_annotation(annotation_path)
            used_annotations.add(annotation_path)
        else:
            pairing = "missing"
        frames.append(
            CaucaFrame(
                source_index=source_index,
                image_path=image_path,
                annotation_path=annotation_path,
                pairing=pairing,
                target=target,
                bbox_xyxy_norm=bbox,
            )
        )

    orphaned = sorted(
        path for path in annotations.values() if path not in used_annotations
    )
    if orphaned:
        rendered = ", ".join(str(path) for path in orphaned[:5])
        raise DatasetLayoutError(
            f"Unmatched annotation files in {activity_root}: {rendered}"
        )
    return tuple(frames)


def discover_videos(dataset_root: str | Path) -> list[CaucaVideo]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetLayoutError(f"Dataset root does not exist: {root}")
    subjects = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=_subject_number,
    )
    if not subjects:
        raise DatasetLayoutError(f"No Subject.* directories found under {root}")

    records: list[CaucaVideo] = []
    for subject_root in subjects:
        subject_id = _subject_number(subject_root)
        activity_roots = sorted(
            (path for path in subject_root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        for activity_root in activity_roots:
            if activity_root.name not in ACTIVITY_LABELS:
                raise DatasetLayoutError(
                    f"Unexpected activity directory: {activity_root}"
                )
            videos = sorted(activity_root.glob("*.avi"))
            if len(videos) != 1:
                raise DatasetLayoutError(
                    f"Expected one AVI in {activity_root}; found {len(videos)}"
                )
            frames = _pair_frames(activity_root)
            label = ACTIVITY_LABELS[activity_root.name]
            observed_label = any(frame.target is True for frame in frames)
            if observed_label != label:
                raise DatasetLayoutError(
                    f"Directory and frame labels disagree in {activity_root}"
                )
            records.append(
                CaucaVideo(
                    sample_id=f"Subject.{subject_id}/{activity_root.name}",
                    subject_id=subject_id,
                    activity=activity_root.name,
                    label=label,
                    lighting=LIGHTING_BY_SUBJECT[subject_id],
                    video_path=videos[0],
                    frames=frames,
                )
            )
    return records


def select_balanced_records(
    records: Sequence[CaucaVideo],
    limit_per_label: int,
) -> list[CaucaVideo]:
    if limit_per_label <= 0:
        return list(records)
    selected: list[CaucaVideo] = []
    counts = {False: 0, True: 0}
    for record in records:
        if counts[record.label] >= limit_per_label:
            continue
        selected.append(record)
        counts[record.label] += 1
    return selected
