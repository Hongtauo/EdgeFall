"""Compare video-level scores on semantically difficult negative actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


OMNIFALL_CATEGORY_IDS = {
    "sit": {3, 4},
    "lie_down": {5, 6},
    "kneel_squat": {10, 11, 12, 13},
}
CAUCAFALL_CATEGORIES = {
    "Sit down": "sit",
    "Kneel": "kneel_squat",
    "Pick up object": "bend_pick",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and candidate video peak scores by negative "
            "action category. Pair syntax is DOMAIN=predictions.csv."
        )
    )
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    return parser


def _parse_pairs(values: Sequence[str]) -> dict[str, Path]:
    pairs: dict[str, Path] = {}
    for value in values:
        domain, separator, path = value.partition("=")
        if not separator or not domain or not path:
            raise ValueError(f"Expected DOMAIN=PATH, got {value!r}")
        if domain in pairs:
            raise ValueError(f"Duplicate domain {domain!r}")
        pairs[domain] = Path(path)
    return pairs


def _load_video_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            video_path = row["path"]
            score = float(row["score"])
            scores[video_path] = max(scores.get(video_path, -math.inf), score)
    return scores


def _omnifall_categories(label_name: str) -> set[str]:
    action_ids = {
        int(value) for value in label_name.split("|") if value.strip()
    }
    return {
        category
        for category, category_ids in OMNIFALL_CATEGORY_IDS.items()
        if action_ids & category_ids
    }


def negative_categories(domain: str, label_name: str) -> set[str]:
    """Return non-exclusive semantic categories for one negative video."""

    if domain == "omnifall":
        categories = _omnifall_categories(label_name)
    elif domain == "caucafall":
        category = CAUCAFALL_CATEGORIES.get(label_name)
        categories = {category} if category else set()
    else:
        categories = set()
    return categories or {"ordinary_other"}


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_pairs(
    rows: Sequence[tuple[float, float]],
    *,
    higher_is_better: bool,
) -> dict[str, float | int]:
    baseline = [row[0] for row in rows]
    candidate = [row[1] for row in rows]
    count = len(rows)
    baseline_mean = sum(baseline) / count
    candidate_mean = sum(candidate) / count
    if higher_is_better:
        improved_count = sum(new > old for old, new in rows)
        worsened_count = sum(new < old for old, new in rows)
    else:
        improved_count = sum(new < old for old, new in rows)
        worsened_count = sum(new > old for old, new in rows)
    return {
        "count": count,
        "preferred_direction": "higher" if higher_is_better else "lower",
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta_mean": candidate_mean - baseline_mean,
        "baseline_median": _quantile(baseline, 0.5),
        "candidate_median": _quantile(candidate, 0.5),
        "baseline_p05": _quantile(baseline, 0.05),
        "candidate_p05": _quantile(candidate, 0.05),
        "baseline_p10": _quantile(baseline, 0.1),
        "candidate_p10": _quantile(candidate, 0.1),
        "baseline_p90": _quantile(baseline, 0.9),
        "candidate_p90": _quantile(candidate, 0.9),
        "baseline_p95": _quantile(baseline, 0.95),
        "candidate_p95": _quantile(candidate, 0.95),
        "improved_count": improved_count,
        "improved_fraction": improved_count / count,
        "worsened_count": worsened_count,
        "worsened_fraction": worsened_count / count,
    }


def build_report(
    manifest_path: Path,
    baseline_paths: Mapping[str, Path],
    candidate_paths: Mapping[str, Path],
    *,
    split: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if baseline_paths.keys() != candidate_paths.keys():
        raise ValueError("Baseline and candidate domains must match")
    domains = set(baseline_paths)
    baseline_scores = {
        domain: _load_video_scores(path)
        for domain, path in baseline_paths.items()
    }
    candidate_scores = {
        domain: _load_video_scores(path)
        for domain, path in candidate_paths.items()
    }

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    details: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            domain = row["domain"]
            if domain not in domains or row["split"] != split:
                continue
            video_path = row["path"]
            baseline_score = baseline_scores[domain].get(video_path, 0.0)
            candidate_score = candidate_scores[domain].get(video_path, 0.0)
            if int(row["label"]) != 0:
                grouped[(domain, "positive_fall")].append(
                    (baseline_score, candidate_score)
                )
                continue
            categories = negative_categories(domain, row["label_name"])
            grouped[(domain, "all_negative")].append(
                (baseline_score, candidate_score)
            )
            for category in categories:
                grouped[(domain, category)].append(
                    (baseline_score, candidate_score)
                )
            details.append(
                {
                    "domain": domain,
                    "path": video_path,
                    "label_name": row["label_name"],
                    "categories": "|".join(sorted(categories)),
                    "baseline_score": baseline_score,
                    "candidate_score": candidate_score,
                    "delta": candidate_score - baseline_score,
                }
            )

    summary: list[dict[str, object]] = []
    for (domain, category), values in sorted(grouped.items()):
        summary.append(
            {
                "domain": domain,
                "category": category,
                **_summarize_pairs(
                    values,
                    higher_is_better=category == "positive_fall",
                ),
            }
        )
    return summary, details


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    baseline = _parse_pairs(args.baseline)
    candidate = _parse_pairs(args.candidate)
    summary, details = build_report(
        args.video_manifest,
        baseline,
        candidate,
        split=args.split,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "categories.csv", summary)
    top_rows = sorted(
        details,
        key=lambda row: float(row["candidate_score"]),
        reverse=True,
    )[: args.top_k]
    _write_csv(args.output_dir / "top_candidate_negative_scores.csv", top_rows)
    payload = {
        "video_manifest": str(args.video_manifest),
        "split": args.split,
        "baseline": {key: str(value) for key, value in baseline.items()},
        "candidate": {key: str(value) for key, value in candidate.items()},
        "categories": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
