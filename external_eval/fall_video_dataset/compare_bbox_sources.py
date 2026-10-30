"""Compare two prediction files on their common evaluable videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Sequence

from external_eval.fall_video_dataset.common import (
    binary_ranking_metrics,
    fixed_threshold_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _metric_values(
    scores: list[float],
    labels: list[bool],
    threshold: float,
) -> dict[str, float]:
    ranking = binary_ranking_metrics(scores, labels)
    fixed = fixed_threshold_metrics(scores, labels, threshold=threshold)
    return {
        "average_precision": float(ranking["average_precision"]),
        "p90": float(ranking["p90"]),
        "p95": float(ranking["p95"]),
        "mean_precision": float(ranking["mean_precision"]),
        "accuracy": float(fixed["accuracy"]),
        "precision": float(fixed["precision"]),
        "recall": float(fixed["recall"]),
        "specificity": float(fixed["specificity"]),
        "f1": float(fixed["f1"]),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _exact_mcnemar_p(reference_only: int, candidate_only: int) -> float:
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    smaller = min(reference_only, candidate_only)
    lower_tail = sum(
        math.comb(discordant, value) for value in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def compare(args: argparse.Namespace) -> Path:
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    reference_rows = _read(args.reference)
    candidate_rows = _read(args.candidate)
    selected = sorted(set(reference_rows) & set(candidate_rows))
    common = [
        sample_id
        for sample_id in selected
        if reference_rows[sample_id]["status"] == "ok"
        and candidate_rows[sample_id]["status"] == "ok"
    ]
    if not common:
        raise ValueError("No common evaluable predictions")
    labels = [
        reference_rows[sample_id]["label"].lower() == "true"
        for sample_id in common
    ]
    reference_scores = [
        float(reference_rows[sample_id]["score"]) for sample_id in common
    ]
    candidate_scores = [
        float(candidate_rows[sample_id]["score"]) for sample_id in common
    ]
    reference_metrics = _metric_values(reference_scores, labels, args.threshold)
    candidate_metrics = _metric_values(candidate_scores, labels, args.threshold)
    deltas = {
        name: candidate_metrics[name] - reference_metrics[name]
        for name in reference_metrics
    }

    bootstrap_deltas = {name: [] for name in reference_metrics}
    randomizer = random.Random(args.seed)
    for _ in range(args.bootstrap_samples):
        indices = [randomizer.randrange(len(common)) for _ in common]
        sample_labels = [labels[index] for index in indices]
        if all(sample_labels) or not any(sample_labels):
            continue
        reference_sample = [reference_scores[index] for index in indices]
        candidate_sample = [candidate_scores[index] for index in indices]
        reference_sample_metrics = _metric_values(
            reference_sample,
            sample_labels,
            args.threshold,
        )
        candidate_sample_metrics = _metric_values(
            candidate_sample,
            sample_labels,
            args.threshold,
        )
        for name in bootstrap_deltas:
            bootstrap_deltas[name].append(
                candidate_sample_metrics[name] - reference_sample_metrics[name]
            )
    intervals = {
        name: {
            "lower_95": _percentile(values, 0.025),
            "upper_95": _percentile(values, 0.975),
        }
        for name, values in bootstrap_deltas.items()
    }

    reference_only = 0
    candidate_only = 0
    same_prediction = 0
    for label, reference_score, candidate_score in zip(
        labels,
        reference_scores,
        candidate_scores,
        strict=True,
    ):
        reference_prediction = reference_score >= args.threshold
        candidate_prediction = candidate_score >= args.threshold
        reference_correct = reference_prediction == label
        candidate_correct = candidate_prediction == label
        reference_only += int(reference_correct and not candidate_correct)
        candidate_only += int(candidate_correct and not reference_correct)
        same_prediction += int(reference_prediction == candidate_prediction)

    score_deltas = [
        candidate - reference
        for reference, candidate in zip(
            reference_scores,
            candidate_scores,
            strict=True,
        )
    ]
    output = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "selected_videos": len(selected),
        "common_evaluable_videos": len(common),
        "reference_not_evaluable": sum(
            reference_rows[sample_id]["status"] != "ok" for sample_id in selected
        ),
        "candidate_not_evaluable": sum(
            candidate_rows[sample_id]["status"] != "ok" for sample_id in selected
        ),
        "threshold": args.threshold,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_reference": deltas,
        "paired_bootstrap_95_percent_interval": intervals,
        "score_delta": {
            "mean": statistics.fmean(score_deltas),
            "median": statistics.median(score_deltas),
            "positive_mean": statistics.fmean(
                value for value, label in zip(score_deltas, labels) if label
            ),
            "negative_mean": statistics.fmean(
                value for value, label in zip(score_deltas, labels) if not label
            ),
        },
        "paired_fixed_threshold_changes": {
            "reference_correct_candidate_wrong": reference_only,
            "reference_wrong_candidate_correct": candidate_only,
            "same_prediction": same_prediction,
            "exact_mcnemar_p": _exact_mcnemar_p(reference_only, candidate_only),
        },
        "bootstrap_samples_requested": args.bootstrap_samples,
        "bootstrap_samples_used": len(
            next(iter(bootstrap_deltas.values()))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return args.output


def main(argv: Sequence[str] | None = None) -> None:
    compare(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
