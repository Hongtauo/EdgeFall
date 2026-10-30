"""Convert OmniFall manual multi-person review into track supervision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


POSITIVE_ACTIONS = frozenset({"fall", "fallen"})
REVIEW_FIELDS = (
    "path",
    "auto_group",
    "primary_track_ids",
    "secondary_track_ids",
    "manual_is_multi",
    "manual_primary_correct",
    "manual_notes",
)
DECISION_FIELDS = ("path", "decision", "target_track_ids", "reason")


@dataclass(frozen=True)
class ReviewDecision:
    path: str
    decision: str
    target_track_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class PersonSupervisionRecord:
    path: str
    action: str
    video_fall_label: int
    manual_is_multi: str
    manual_primary_correct: str
    primary_track_ids: str
    secondary_track_ids: str
    target_track_ids: str
    negative_track_ids: str
    action_target_track_ids: str
    use_in_training: int
    use_in_person_supervision: int
    disposition: str
    reason: str


def _track_ids(value: str, *, field: str, path: str) -> tuple[int, ...]:
    stripped = value.strip()
    if not stripped:
        return ()
    values: set[int] = set()
    for item in stripped.split("|"):
        try:
            track_id = int(item)
        except ValueError as error:
            raise ValueError(
                f"Invalid {field} track ID for {path}: {item!r}"
            ) from error
        if track_id < 0:
            raise ValueError(f"Negative {field} track ID for {path}")
        values.add(track_id)
    return tuple(sorted(values))


def _format_track_ids(values: Iterable[int]) -> str:
    return "|".join(str(value) for value in sorted(set(values)))


def _read_csv(
    path: Path,
    required_fields: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(set(required_fields).difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing fields {missing}: {path}")
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows and not allow_empty:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def load_decisions(path: Path) -> dict[str, ReviewDecision]:
    decisions: dict[str, ReviewDecision] = {}
    for row in _read_csv(path, DECISION_FIELDS, allow_empty=True):
        sample_path = row["path"]
        if sample_path in decisions:
            raise ValueError(f"Duplicate decision path: {sample_path}")
        decision = row["decision"]
        if decision not in {"use", "exclude"}:
            raise ValueError(
                f"Unsupported decision for {sample_path}: {decision!r}"
            )
        target_ids = _track_ids(
            row["target_track_ids"],
            field="decision target",
            path=sample_path,
        )
        if decision == "use" and not target_ids:
            raise ValueError(f"Use decision requires target IDs: {sample_path}")
        if decision == "exclude" and target_ids:
            raise ValueError(
                f"Exclude decision cannot contain target IDs: {sample_path}"
            )
        decisions[sample_path] = ReviewDecision(
            path=sample_path,
            decision=decision,
            target_track_ids=target_ids,
            reason=row["reason"],
        )
    return decisions


def _record(
    row: dict[str, str],
    decision: ReviewDecision | None,
) -> PersonSupervisionRecord:
    sample_path = row["path"]
    action = sample_path.split("/", 1)[0]
    positive = action in POSITIVE_ACTIONS
    is_multi = row["manual_is_multi"]
    primary_correct = row["manual_primary_correct"]
    if is_multi not in {"yes", "no", "uncertain"}:
        raise ValueError(f"Invalid manual_is_multi for {sample_path}")
    if primary_correct not in {"yes", "no", "uncertain"}:
        raise ValueError(f"Invalid manual_primary_correct for {sample_path}")

    primary_ids = _track_ids(
        row["primary_track_ids"], field="primary", path=sample_path
    )
    secondary_ids = _track_ids(
        row["secondary_track_ids"], field="secondary", path=sample_path
    )
    if not primary_ids:
        raise ValueError(f"Review row has no primary IDs: {sample_path}")

    target_ids: tuple[int, ...] = ()
    negative_ids: tuple[int, ...] = ()
    action_target_ids: tuple[int, ...] = ()
    use_in_training = 1
    use_in_person_supervision = 1
    reason = ""

    if decision is not None and decision.decision == "exclude":
        if is_multi != "yes" or not positive:
            raise ValueError(
                f"Only reviewed multi-person positives may be excluded: {sample_path}"
            )
        use_in_training = 0
        use_in_person_supervision = 0
        disposition = "excluded_unresolved_positive"
        reason = decision.reason
    elif is_multi == "yes" and positive:
        if primary_correct == "yes":
            if decision is not None:
                raise ValueError(
                    f"Verified primary does not need a decision: {sample_path}"
                )
            target_ids = primary_ids
            disposition = "multi_positive_primary_verified"
        elif decision is not None and decision.decision == "use":
            target_ids = decision.target_track_ids
            disposition = "multi_positive_target_corrected"
            reason = decision.reason
        else:
            raise ValueError(
                f"Unresolved multi-person positive requires a decision: {sample_path}"
            )
        all_ids = set(primary_ids) | set(secondary_ids) | set(target_ids)
        negative_ids = tuple(sorted(all_ids.difference(target_ids)))
        action_target_ids = target_ids
    elif is_multi == "yes":
        if decision is not None:
            raise ValueError(f"Unexpected decision for negative: {sample_path}")
        all_ids = tuple(sorted(set(primary_ids) | set(secondary_ids)))
        negative_ids = all_ids
        action_target_ids = primary_ids if primary_correct == "yes" else ()
        if secondary_ids:
            disposition = "multi_negative_all_tracks"
        else:
            disposition = "multi_negative_partial_tracks"
            reason = "reviewed_multi_but_secondary_track_ids_are_empty"
    elif is_multi == "no":
        if decision is not None:
            raise ValueError(f"Unexpected decision for non-multi: {sample_path}")
        if positive and primary_correct == "yes":
            target_ids = primary_ids
            action_target_ids = primary_ids
            disposition = "single_positive_primary_verified"
        elif positive:
            use_in_person_supervision = 0
            disposition = "single_positive_primary_unresolved"
            reason = "retain_base_training_without_person_supervision"
        else:
            negative_ids = primary_ids
            action_target_ids = primary_ids if primary_correct == "yes" else ()
            disposition = "single_negative"
    else:
        if decision is not None:
            raise ValueError(f"Unexpected decision for uncertain row: {sample_path}")
        use_in_person_supervision = 0
        disposition = "multi_status_uncertain"
        reason = "retain_base_training_without_multi_person_supervision"

    return PersonSupervisionRecord(
        path=sample_path,
        action=action,
        video_fall_label=int(positive),
        manual_is_multi=is_multi,
        manual_primary_correct=primary_correct,
        primary_track_ids=_format_track_ids(primary_ids),
        secondary_track_ids=_format_track_ids(secondary_ids),
        target_track_ids=_format_track_ids(target_ids),
        negative_track_ids=_format_track_ids(negative_ids),
        action_target_track_ids=_format_track_ids(action_target_ids),
        use_in_training=use_in_training,
        use_in_person_supervision=use_in_person_supervision,
        disposition=disposition,
        reason=reason,
    )


def build_supervision(
    review_path: Path,
    decisions_path: Path,
) -> list[PersonSupervisionRecord]:
    review_rows = _read_csv(review_path, REVIEW_FIELDS)
    decisions = load_decisions(decisions_path)
    review_by_path: dict[str, dict[str, str]] = {}
    for row in review_rows:
        sample_path = row["path"]
        if sample_path in review_by_path:
            raise ValueError(f"Duplicate review path: {sample_path}")
        review_by_path[sample_path] = row
    unknown = sorted(set(decisions).difference(review_by_path))
    if unknown:
        raise ValueError(f"Decisions reference unknown paths: {unknown}")
    return [
        _record(row, decisions.get(row["path"])) for row in review_rows
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_supervision(
    records: list[PersonSupervisionRecord],
    *,
    output_path: Path,
    summary_path: Path,
    review_path: Path,
    decisions_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    disposition_counts = Counter(record.disposition for record in records)
    action_counts = Counter(record.action for record in records)
    summary: dict[str, object] = {
        "review": str(review_path),
        "review_sha256": _sha256(review_path),
        "decisions": str(decisions_path),
        "decisions_sha256": _sha256(decisions_path),
        "output": str(output_path),
        "counts": {
            "records": len(records),
            "use_in_training": sum(record.use_in_training for record in records),
            "excluded_from_training": sum(
                not record.use_in_training for record in records
            ),
            "use_in_person_supervision": sum(
                record.use_in_person_supervision for record in records
            ),
            "manual_multi_yes": sum(
                record.manual_is_multi == "yes" for record in records
            ),
            "usable_multi_positive": sum(
                record.manual_is_multi == "yes"
                and record.video_fall_label == 1
                and record.use_in_training == 1
                for record in records
            ),
            "multi_negative": sum(
                record.manual_is_multi == "yes"
                and record.video_fall_label == 0
                for record in records
            ),
        },
        "dispositions": dict(sorted(disposition_counts.items())),
        "actions": dict(sorted(action_counts.items())),
        "excluded_paths": [
            record.path for record in records if not record.use_in_training
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(
            "annotations/omnifall/train_multi_person_manual_review.csv"
        ),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "annotations/omnifall/train_multi_person_decisions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/reviews/omnifall_train_multi_person_supervision.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "artifacts/reviews/"
            "omnifall_train_multi_person_supervision_summary.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_supervision(args.review, args.decisions)
    summary = write_supervision(
        records,
        output_path=args.output,
        summary_path=args.summary,
        review_path=args.review,
        decisions_path=args.decisions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
