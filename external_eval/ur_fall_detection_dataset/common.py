"""Dataset discovery for UR Fall Detection RGB frame sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EXPECTED_FALL_SEQUENCES = 30
EXPECTED_ADL_SEQUENCES = 40
SEQUENCE_PATTERN = re.compile(r"(fall|adl)-(\d{2})-cam0-rgb", re.IGNORECASE)


class DatasetLayoutError(RuntimeError):
    """Raised when the local URFD copy does not match the expected layout."""


@dataclass(frozen=True)
class UrFallSequence:
    sample_id: str
    label: bool
    frame_paths: tuple[Path, ...]


def _sequence_root(path: Path) -> Path:
    nested = path / "UR_fall_detection_dataset_cam0_rgb"
    return nested if nested.is_dir() else path


def _frame_number(path: Path, sequence_name: str) -> int:
    match = re.fullmatch(
        rf"{re.escape(sequence_name)}-(\d+)\.png",
        path.name,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise DatasetLayoutError(f"Unexpected frame name: {path}")
    return int(match.group(1))


def discover_sequences(
    dataset_root: str | Path,
    *,
    require_complete: bool = True,
) -> list[UrFallSequence]:
    root = _sequence_root(Path(dataset_root))
    if not root.is_dir():
        raise DatasetLayoutError(f"Dataset root does not exist: {root}")

    records: list[UrFallSequence] = []
    for sequence_root in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        match = SEQUENCE_PATTERN.fullmatch(sequence_root.name)
        if match is None:
            raise DatasetLayoutError(
                f"Unexpected sequence directory: {sequence_root}"
            )
        numbered = sorted(
            (
                (_frame_number(path, sequence_root.name), path)
                for path in sequence_root.glob("*.png")
            ),
            key=lambda item: item[0],
        )
        if not numbered:
            raise DatasetLayoutError(f"No PNG frames found in {sequence_root}")
        frame_numbers = [number for number, _ in numbered]
        expected = list(range(1, len(numbered) + 1))
        if frame_numbers != expected:
            raise DatasetLayoutError(
                f"Non-contiguous frame numbers in {sequence_root}"
            )
        records.append(
            UrFallSequence(
                sample_id=sequence_root.name,
                label=match.group(1).casefold() == "fall",
                frame_paths=tuple(path for _, path in numbered),
            )
        )

    if require_complete:
        fall_count = sum(record.label for record in records)
        adl_count = len(records) - fall_count
        if (fall_count, adl_count) != (
            EXPECTED_FALL_SEQUENCES,
            EXPECTED_ADL_SEQUENCES,
        ):
            raise DatasetLayoutError(
                "Expected 30 fall and 40 ADL sequences; "
                f"found {fall_count} fall and {adl_count} ADL"
            )
    return records
