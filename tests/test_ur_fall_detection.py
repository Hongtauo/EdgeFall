from pathlib import Path

import pytest

from external_eval.ur_fall_detection_dataset.common import (
    DatasetLayoutError,
    discover_sequences,
)


def _sequence(root: Path, name: str, frame_numbers: tuple[int, ...]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    for frame_number in frame_numbers:
        (directory / f"{name}-{frame_number:03d}.png").touch()


def test_discover_complete_urfd_layout(tmp_path: Path) -> None:
    root = tmp_path / "UR_fall_detection_dataset_cam0_rgb"
    for prefix, count in (("fall", 30), ("adl", 40)):
        for sequence_id in range(1, count + 1):
            name = f"{prefix}-{sequence_id:02d}-cam0-rgb"
            _sequence(root, name, (1, 2))

    records = discover_sequences(tmp_path)

    assert len(records) == 70
    assert sum(record.label for record in records) == 30
    assert all(len(record.frame_paths) == 2 for record in records)


def test_discover_rejects_non_contiguous_frames(tmp_path: Path) -> None:
    name = "fall-01-cam0-rgb"
    _sequence(tmp_path, name, (1, 3))

    with pytest.raises(DatasetLayoutError, match="Non-contiguous"):
        discover_sequences(tmp_path, require_complete=False)
