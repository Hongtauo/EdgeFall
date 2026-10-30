"""Render existing pseudo tracks on their source videos for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgefall_st.pseudo.yolo26_tracker import render_track_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Dataset-relative video paths without the .mp4 suffix.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    parser.add_argument(
        "--tracks-root",
        type=Path,
        default=Path("artifacts/pseudo_labels/pilot_yolo26x_nms/tracks"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/pseudo_labels/pilot_yolo26x_nms/review_visualizations"
        ),
    )
    parser.add_argument(
        "--actor-targets-root",
        type=Path,
        help="Optional root containing per-video actor target JSONL files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for relative_path in args.paths:
        video_path = args.dataset_root / f"{relative_path}.mp4"
        tracks_path = args.tracks_root / f"{relative_path}.jsonl"
        output_path = args.output_root / f"{relative_path}.mp4"
        actor_targets_path = (
            args.actor_targets_root / f"{relative_path}.jsonl"
            if args.actor_targets_root is not None
            else None
        )
        frames = render_track_visualization(
            video_path,
            tracks_path,
            output_path,
            actor_targets_path,
        )
        print(f"{relative_path}: rendered {frames} frames -> {output_path}")


if __name__ == "__main__":
    main()
