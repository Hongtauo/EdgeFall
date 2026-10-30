"""Build leakage-aware V3.5 multi-domain train/validation/test manifests."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, asdict
from pathlib import Path

from external_eval.caucafall.common import CaucaVideo, discover_videos
from external_eval.fall_video_dataset.common import VideoRecord, discover_records


DEFAULT_SEED = 20260727


@dataclass(frozen=True)
class MultiDomainRow:
    domain: str
    split: str
    path: str
    label: int
    label_name: str
    group_id: str
    subject_id: str
    video_path: str
    annotation_source: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_exact_groups(
    rows: Sequence[object],
    count: int,
    *,
    group_key: Callable[[object], str],
    randomizer: random.Random,
) -> tuple[list[object], list[object]]:
    """Select an exact row count without splitting source groups.

    Singleton groups are held back until the end so they can satisfy the exact
    requested count after larger paired/session groups have been selected.
    """

    if not 0 <= count <= len(rows):
        raise ValueError("Requested grouped sample count is out of range")
    grouped: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    groups = list(grouped.values())
    randomizer.shuffle(groups)
    singleton = [group for group in groups if len(group) == 1]
    multiple = [group for group in groups if len(group) > 1]
    randomizer.shuffle(singleton)
    randomizer.shuffle(multiple)

    selected_groups: list[list[object]] = []
    selected_count = 0
    singleton_capacity = len(singleton)
    for group in multiple:
        size = len(group)
        remaining_after = count - (selected_count + size)
        if remaining_after < 0 or remaining_after > singleton_capacity:
            continue
        selected_groups.append(group)
        selected_count += size
    needed_singletons = count - selected_count
    if needed_singletons < 0 or needed_singletons > len(singleton):
        raise RuntimeError(
            "Could not satisfy exact sample count without splitting groups"
        )
    selected_groups.extend(singleton[:needed_singletons])
    selected_ids = {id(row) for group in selected_groups for row in group}
    selected = [row for row in rows if id(row) in selected_ids]
    remaining = [row for row in rows if id(row) not in selected_ids]
    if len(selected) != count:
        raise RuntimeError("Grouped sampling returned an incorrect count")
    return selected, remaining


def _omnifall_rows(
    manifest: Path,
    *,
    train_count: int,
    randomizer: random.Random,
) -> list[MultiDomainRow]:
    source = _read_csv(manifest)
    train = [row for row in source if row["split"] == "train"]
    selected, _ = _sample_exact_groups(
        train,
        train_count,
        group_key=lambda row: str(row["path"]),
        randomizer=randomizer,
    )
    output: list[MultiDomainRow] = []
    for row in selected:
        labels = {int(value) for value in str(row["labels"]).split("|")}
        output.append(
            MultiDomainRow(
                domain="omnifall",
                split="train",
                path=str(row["path"]),
                label=int(bool(labels.intersection((1, 2)))),
                label_name=str(row["labels"]),
                group_id=str(row["path"]),
                subject_id="",
                video_path="",
                annotation_source="omnifall_frame",
            )
        )
    for split in ("val", "test"):
        for row in source:
            if row["split"] != split:
                continue
            labels = {int(value) for value in str(row["labels"]).split("|")}
            output.append(
                MultiDomainRow(
                    domain="omnifall",
                    split=split,
                    path=str(row["path"]),
                    label=int(bool(labels.intersection((1, 2)))),
                    label_name=str(row["labels"]),
                    group_id=str(row["path"]),
                    subject_id="",
                    video_path="",
                    annotation_source="omnifall_frame",
                )
            )
    return output


def _fall_rows(
    records: Sequence[VideoRecord],
    *,
    train_count: int,
    val_count: int,
    randomizer: random.Random,
) -> list[MultiDomainRow]:
    train, remaining = _sample_exact_groups(
        records,
        train_count,
        group_key=lambda row: str(row.group_id),
        randomizer=randomizer,
    )
    val, test = _sample_exact_groups(
        remaining,
        val_count,
        group_key=lambda row: str(row.group_id),
        randomizer=randomizer,
    )
    split_by_id = {
        id(row): split
        for split, values in (("train", train), ("val", val), ("test", test))
        for row in values
    }
    return [
        MultiDomainRow(
            domain="fall_video_dataset",
            split=split_by_id[id(record)],
            path=f"fall_video_dataset/{record.sample_id}",
            label=int(record.label),
            label_name=record.label_name,
            group_id=record.group_id,
            subject_id="",
            video_path=str(record.video_path.resolve()),
            annotation_source="video_binary",
        )
        for record in records
    ]


def _sample_cauca_training(
    records: Sequence[CaucaVideo],
    count: int,
    randomizer: random.Random,
) -> list[CaucaVideo]:
    """Select exact CAUCA training videos while covering labels/activities."""

    if not 0 <= count <= len(records):
        raise ValueError("CAUCA training count is out of range")
    selected: list[CaucaVideo] = []
    positive_count = (count + 1) // 2
    for label, label_count in ((True, positive_count), (False, count - positive_count)):
        by_activity: dict[str, list[CaucaVideo]] = defaultdict(list)
        for record in records:
            if record.label == label:
                by_activity[record.activity].append(record)
        for values in by_activity.values():
            randomizer.shuffle(values)
        activities = sorted(by_activity)
        cursor = 0
        label_selected = 0
        while label_selected < label_count:
            activity = activities[cursor % len(activities)]
            values = by_activity[activity]
            if values:
                selected.append(values.pop())
                label_selected += 1
            cursor += 1
            if cursor > label_count * max(len(activities), 1) * 2:
                raise RuntimeError(
                    "Could not sample the requested CAUCA label balance"
                )
    randomizer.shuffle(selected)
    return selected


def _cauca_rows(
    records: Sequence[CaucaVideo],
    *,
    train_count: int,
    randomizer: random.Random,
) -> list[MultiDomainRow]:
    # This fixed protocol gives training all three lighting domains while
    # retaining subject-disjoint validation and test sets.
    train_subjects = {1, 2, 5, 6, 7, 9}
    val_subjects = {3, 10}
    test_subjects = {4, 8}
    train_pool = [row for row in records if row.subject_id in train_subjects]
    selected_train = _sample_cauca_training(
        train_pool, train_count, randomizer
    )
    selected_ids = {id(row) for row in selected_train}
    output: list[MultiDomainRow] = []
    for record in records:
        if id(record) in selected_ids:
            split = "train"
        elif record.subject_id in val_subjects:
            split = "val"
        elif record.subject_id in test_subjects:
            split = "test"
        else:
            continue
        output.append(
            MultiDomainRow(
                domain="caucafall",
                split=split,
                path=f"caucafall/{record.sample_id}",
                label=int(record.label),
                label_name=record.activity,
                group_id=f"Subject.{record.subject_id}",
                subject_id=str(record.subject_id),
                video_path=str(record.video_path.resolve()),
                annotation_source="frame_binary",
            )
        )
    return output


def build_manifest(args: argparse.Namespace) -> Path:
    randomizer = random.Random(args.seed)
    rows = _omnifall_rows(
        args.omnifall_manifest,
        train_count=args.omnifall_train_count,
        randomizer=randomizer,
    )
    rows.extend(
        _fall_rows(
            discover_records(args.fall_root),
            train_count=args.fall_train_count,
            val_count=args.fall_val_count,
            randomizer=randomizer,
        )
    )
    rows.extend(
        _cauca_rows(
            discover_videos(args.cauca_root),
            train_count=args.cauca_train_count,
            randomizer=randomizer,
        )
    )
    rows.sort(key=lambda row: (row.domain, row.split, row.path.casefold()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    fields = tuple(MultiDomainRow.__dataclass_fields__)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    temporary.replace(args.output)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row.domain, row.split)] += 1
    for key in sorted(counts):
        print(f"{key[0]} {key[1]}={counts[key]}", flush=True)
    print(f"manifest={args.output}", flush=True)
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--omnifall-manifest",
        type=Path,
        default=Path("artifacts/omnifall_v2/split_manifest.csv"),
    )
    parser.add_argument(
        "--fall-root",
        type=Path,
        default=Path("Dataset/fall-video-dataset"),
    )
    parser.add_argument(
        "--cauca-root",
        type=Path,
        default=Path("Dataset/caucafall/Dataset CAUCAFall/CAUCAFall"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multidomain_cauca60/split_manifest.csv"),
    )
    parser.add_argument("--omnifall-train-count", type=int, default=6000)
    parser.add_argument("--fall-train-count", type=int, default=1048)
    parser.add_argument("--fall-val-count", type=int, default=699)
    parser.add_argument("--cauca-train-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    build_manifest(build_parser().parse_args())


if __name__ == "__main__":
    main()
