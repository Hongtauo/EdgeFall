"""Audit the standalone Fall/No_Fall dataset without modifying it."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from external_eval.fall_video_dataset.common import discover_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/fall-video-dataset"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def _has_pose_rows(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return next(reader, None) is not None


def audit(dataset_root: Path) -> dict[str, object]:
    records = discover_records(dataset_root)
    class_counts = Counter(record.label_name for record in records)
    empty_csv = [
        record.sample_id
        for record in records
        if not _has_pose_rows(record.keypoints_path)
    ]
    group_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_labels[record.group_id].add(record.label_name)
    cross_label_groups = sorted(
        group_id for group_id, labels in group_labels.items() if len(labels) > 1
    )
    return {
        "dataset_root": str(dataset_root.resolve()),
        "total_records": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "empty_keypoint_csv_count": len(empty_csv),
        "empty_keypoint_samples": empty_csv,
        "unique_group_count": len(group_labels),
        "cross_label_group_count": len(cross_label_groups),
        "cross_label_groups": cross_label_groups,
        "group_definition": "lowercase stem with one trailing _resized removed",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit(args.dataset_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
