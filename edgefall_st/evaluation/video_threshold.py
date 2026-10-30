"""Lock one video alarm threshold on validation predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class VideoRecord:
    domain: str
    path: str
    label: bool
    score: float


@dataclass(frozen=True)
class ClassificationMetrics:
    videos: int
    positives: int
    negatives: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    f1: float


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    records: Sequence[VideoRecord], threshold: float
) -> ClassificationMetrics:
    tp = fp = tn = fn = 0
    for record in records:
        prediction = record.score >= threshold
        if prediction and record.label:
            tp += 1
        elif prediction:
            fp += 1
        elif record.label:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return ClassificationMetrics(
        videos=len(records),
        positives=tp + fn,
        negatives=tn + fp,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=_ratio(tp + tn, len(records)),
        precision=precision,
        recall=recall,
        specificity=specificity,
        balanced_accuracy=0.5 * (recall + specificity),
        f1=_ratio(2.0 * precision * recall, precision + recall),
    )


def select_f1_threshold(records: Sequence[VideoRecord]) -> float:
    """Maximize pooled validation F1 without interpolating score levels."""

    if not records:
        raise ValueError("Validation records cannot be empty")
    positives = sum(int(record.label) for record in records)
    if positives == 0:
        raise ValueError("Validation records contain no positive videos")
    ordered = sorted(records, key=lambda record: record.score, reverse=True)
    tp = fp = 0
    best_key: tuple[float, float, float, float] | None = None
    best_threshold = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index].score
        group_tp = group_size = 0
        while index < len(ordered) and ordered[index].score == score:
            group_tp += int(ordered[index].label)
            group_size += 1
            index += 1
        tp += group_tp
        fp += group_size - group_tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, positives)
        f1 = _ratio(2.0 * precision * recall, precision + recall)
        key = (f1, recall, precision, score)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = score
    return float(best_threshold)


def _mapping(value: str) -> tuple[str, Path]:
    domain, separator, path = value.partition("=")
    if not separator or not domain.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected DOMAIN=PREDICTIONS.csv")
    return domain.strip(), Path(path.strip())


def load_video_records(
    manifest_path: Path,
    predictions_path: Path,
    *,
    domain: str,
    split: str,
) -> tuple[list[VideoRecord], int]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row.get("domain") == domain and row.get("split") == split
        ]
    expected = {str(row["path"]): bool(int(row["label"])) for row in selected}
    if not expected:
        raise ValueError(f"No manifest videos for {domain}/{split}")
    if len(expected) != len(selected):
        raise ValueError(f"Duplicate manifest paths for {domain}/{split}")

    maxima: dict[str, float] = {}
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = str(row["path"])
            if path not in expected:
                raise ValueError(
                    f"Prediction path is absent from {domain}/{split}: {path}"
                )
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score for {path}")
            maxima[path] = max(maxima.get(path, 0.0), score)
    records = [
        VideoRecord(
            domain=domain,
            path=path,
            label=label,
            score=maxima.get(path, 0.0),
        )
        for path, label in expected.items()
    ]
    return records, len(expected) - len(maxima)


def _metrics_by_domain(
    records_by_domain: dict[str, list[VideoRecord]], threshold: float
) -> dict[str, Any]:
    pooled = [
        record
        for records in records_by_domain.values()
        for record in records
    ]
    return {
        "pooled": asdict(classification_metrics(pooled, threshold)),
        "domains": {
            domain: asdict(classification_metrics(records, threshold))
            for domain, records in sorted(records_by_domain.items())
        },
    }


def calibrate(
    *,
    manifest_path: Path,
    validation_predictions: Sequence[tuple[str, Path]],
    test_predictions: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    validation: dict[str, list[VideoRecord]] = {}
    missing_validation: dict[str, int] = {}
    for domain, path in validation_predictions:
        if domain in validation:
            raise ValueError(f"Duplicate validation domain: {domain}")
        validation[domain], missing_validation[domain] = load_video_records(
            manifest_path, path, domain=domain, split="val"
        )
    pooled_validation = [
        record for records in validation.values() for record in records
    ]
    threshold = select_f1_threshold(pooled_validation)

    # Test predictions are intentionally loaded only after threshold selection.
    test: dict[str, list[VideoRecord]] = {}
    missing_test: dict[str, int] = {}
    for domain, path in test_predictions:
        if domain in test:
            raise ValueError(f"Duplicate test domain: {domain}")
        test[domain], missing_test[domain] = load_video_records(
            manifest_path, path, domain=domain, split="test"
        )
    return {
        "selection": {
            "source": "validation_only",
            "objective": "pooled_video_f1",
            "tie_break": ["recall", "precision", "threshold"],
            "threshold": threshold,
        },
        "validation": {
            **_metrics_by_domain(validation, threshold),
            "missing_videos": missing_validation,
        },
        "test": {
            **_metrics_by_domain(test, threshold),
            "missing_videos": missing_test,
        },
        "inputs": {
            "manifest": str(manifest_path),
            "validation_predictions": {
                domain: str(path) for domain, path in validation_predictions
            },
            "test_predictions": {
                domain: str(path) for domain, path in test_predictions
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-predictions",
        type=_mapping,
        action="append",
        required=True,
        metavar="DOMAIN=CSV",
    )
    parser.add_argument(
        "--test-predictions",
        type=_mapping,
        action="append",
        required=True,
        metavar="DOMAIN=CSV",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = calibrate(
        manifest_path=args.manifest,
        validation_predictions=args.validation_predictions,
        test_predictions=args.test_predictions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
