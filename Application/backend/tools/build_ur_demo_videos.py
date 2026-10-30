"""Convert selected UR Fall Detection Dataset RGB sequences into demo MP4 files."""

from __future__ import annotations

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = (
    ROOT
    / "Dataset"
    / "ur-fall-detection-dataset"
    / "UR_fall_detection_dataset_cam0_rgb"
)
OUTPUT_ROOT = ROOT / "demo" / "source"
SOURCE_FPS = 30.0

SEQUENCES = {
    "fall-01-cam0-rgb": "15_urfd_side_fall.mp4",
    "fall-20-cam0-rgb": "16_urfd_chair_fall.mp4",
    "adl-06-cam0-rgb": "17_urfd_bending_walk.mp4",
    "adl-24-cam0-rgb": "18_urfd_tie_shoe.mp4",
    "adl-10-cam0-rgb": "23_urfd_daily_walk.mp4",
}


def convert(sequence_name: str, output_name: str) -> Path:
    frame_paths = sorted((DATASET_ROOT / sequence_name).glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No RGB frames found for {sequence_name}")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Could not read {frame_paths[0]}")
    height, width = first.shape[:2]
    output_path = OUTPUT_ROOT / output_name
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), SOURCE_FPS, (width, height)
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create {output_path}")
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Invalid frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sequence_name, output_name in SEQUENCES.items():
        output = convert(sequence_name, output_name)
        print(f"created={output}", flush=True)


if __name__ == "__main__":
    main()
