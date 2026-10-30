"""Audit video-level fall-classification failures across model versions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from edgefall_st.evaluation.video_threshold import (
    VideoRecord,
    classification_metrics,
)


ACTION_NAMES = {
    0: "walk",
    1: "fall",
    2: "fallen",
    3: "sit_down",
    4: "sitting",
    5: "lie_down",
    6: "lying",
    7: "stand_up",
    8: "standing",
    9: "other",
    10: "kneel_down",
    11: "kneeling",
    12: "squat_down",
    13: "squatting",
    14: "crawl",
    15: "jump",
}


@dataclass(frozen=True)
class ManifestVideo:
    domain: str
    path: str
    label: bool
    label_name: str
    video_path: Path | None


@dataclass(frozen=True)
class WindowPrediction:
    window_index: int
    end_time_s: float
    score: float
    geometry_score: float
    global_score: float
    roi_score: float
    stage_label: int
    action_label: int
    global_attention: float
    roi_attention: float


@dataclass(frozen=True)
class VideoPrediction:
    peak: WindowPrediction
    geometry_video_score: float
    global_video_score: float
    roi_video_score: float
    series: tuple[WindowPrediction, ...]


def _mapping(value: str) -> tuple[str, Path]:
    domain, separator, path = value.partition("=")
    if not separator or not domain.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected DOMAIN=PREDICTIONS.csv")
    return domain.strip(), Path(path.strip())


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {None, ""}:
        return default
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {key}: {value}")
    return number


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value in {None, ""} else int(float(value))


def load_manifest(
    path: Path,
    *,
    split: str,
    domains: set[str],
    video_roots: dict[str, Path] | None = None,
) -> dict[str, dict[str, ManifestVideo]]:
    video_roots = video_roots or {}
    selected: dict[str, dict[str, ManifestVideo]] = {
        domain: {} for domain in domains
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            domain = str(row.get("domain", ""))
            if domain not in domains or row.get("split") != split:
                continue
            video_path_text = str(row.get("video_path", "")).strip()
            video_path = Path(video_path_text) if video_path_text else None
            if video_path is None and domain in video_roots:
                relative_path = Path(str(row["path"]).replace("\\", "/"))
                candidates = [
                    video_roots[domain] / relative_path.with_suffix(extension)
                    for extension in (".mp4", ".avi", ".mov", ".mkv")
                ]
                video_path = next(
                    (candidate for candidate in candidates if candidate.is_file()),
                    candidates[0],
                )
            record = ManifestVideo(
                domain=domain,
                path=str(row["path"]),
                label=bool(int(row["label"])),
                label_name=str(row.get("label_name", "")),
                video_path=video_path,
            )
            if record.path in selected[domain]:
                raise ValueError(f"Duplicate manifest path: {record.path}")
            selected[domain][record.path] = record
    for domain, records in selected.items():
        if not records:
            raise ValueError(f"No manifest videos for {domain}/{split}")
    return selected


def load_predictions(
    path: Path,
    *,
    expected_paths: set[str],
) -> tuple[dict[str, VideoPrediction], int]:
    rows_by_video: dict[str, list[WindowPrediction]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            video_path = str(raw["path"])
            if video_path not in expected_paths:
                raise ValueError(f"Prediction path is not in manifest: {video_path}")
            row = WindowPrediction(
                window_index=_int(raw, "window_index"),
                end_time_s=_float(raw, "end_time_s"),
                score=_float(raw, "score"),
                geometry_score=_float(raw, "geometry_score"),
                global_score=_float(raw, "global_score"),
                roi_score=_float(raw, "roi_score"),
                stage_label=_int(raw, "stage_label"),
                action_label=_int(raw, "action_label", 9),
                global_attention=_float(raw, "global_attention"),
                roi_attention=_float(raw, "roi_attention"),
            )
            rows_by_video.setdefault(video_path, []).append(row)

    predictions: dict[str, VideoPrediction] = {}
    for video_path, rows in rows_by_video.items():
        series = tuple(sorted(rows, key=lambda row: (row.end_time_s, row.window_index)))
        peak = max(series, key=lambda row: (row.score, -row.window_index))
        predictions[video_path] = VideoPrediction(
            peak=peak,
            geometry_video_score=max(row.geometry_score for row in series),
            global_video_score=max(row.global_score for row in series),
            roi_video_score=max(row.roi_score for row in series),
            series=series,
        )
    return predictions, len(expected_paths - predictions.keys())


def precision_at_recall(
    predictions: dict[str, VideoPrediction],
    manifest: dict[str, ManifestVideo],
    target: float,
) -> dict[str, float | int]:
    scores = {
        path: predictions[path].peak.score if path in predictions else 0.0
        for path in manifest
    }
    return precision_at_recall_scores(scores, manifest, target)


def precision_at_recall_scores(
    scores: dict[str, float],
    manifest: dict[str, ManifestVideo],
    target: float,
) -> dict[str, float | int]:
    ordered = sorted(
        (
            (scores.get(path, 0.0), record.label)
            for path, record in manifest.items()
        ),
        reverse=True,
    )
    positive_count = sum(int(record.label) for record in manifest.values())
    tp = fp = index = 0
    best: dict[str, float | int] | None = None
    while index < len(ordered):
        score = ordered[index][0]
        group_tp = group_size = 0
        while index < len(ordered) and ordered[index][0] == score:
            group_tp += int(ordered[index][1])
            group_size += 1
            index += 1
        tp += group_tp
        fp += group_size - group_tp
        recall = tp / positive_count if positive_count else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        point = {
            "threshold": score,
            "tp": tp,
            "fp": fp,
            "fn": positive_count - tp,
            "precision": precision,
            "recall": recall,
        }
        if recall >= target and (
            best is None
            or (precision, score)
            > (float(best["precision"]), float(best["threshold"]))
        ):
            best = point
    if best is None:
        raise ValueError(f"Could not reach recall={target}")
    return best


def _persistent_score(
    prediction: VideoPrediction, run_length: int
) -> float:
    scores = [row.score for row in prediction.series]
    if run_length <= 1:
        return max(scores, default=0.0)
    if len(scores) < run_length:
        return 0.0
    return max(
        min(scores[index : index + run_length])
        for index in range(len(scores) - run_length + 1)
    )


def _semantic_group(domain: str, path: str, label_name: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if domain == "fall_video_dataset" and len(parts) > 1:
        return parts[1]
    if domain == "caucafall":
        return label_name or parts[-1]
    return parts[0]


def _classification(label: bool, alarm: bool) -> str:
    if alarm and label:
        return "tp"
    if alarm:
        return "fp"
    if label:
        return "fn"
    return "tn"


def _transition(baseline_error: str, candidate_error: str) -> str:
    baseline_correct = baseline_error in {"tp", "tn"}
    candidate_correct = candidate_error in {"tp", "tn"}
    if baseline_correct and candidate_correct:
        return "both_correct"
    if baseline_correct:
        return "baseline_only_correct"
    if candidate_correct:
        return "candidate_only_correct"
    return "both_wrong"


def _operating_transition(
    *,
    label: bool,
    baseline_alarm: bool,
    candidate_alarm: bool,
) -> str:
    suffix = "fn" if label else "fp"
    baseline_error = baseline_alarm != label
    candidate_error = candidate_alarm != label
    if candidate_error and not baseline_error:
        return f"new_{suffix}"
    if candidate_error and baseline_error:
        return f"common_{suffix}"
    if baseline_error:
        return f"rescued_{suffix}"
    return "correct"


def _fusion_crossing(final_alarm: bool, geometry_alarm: bool) -> str:
    if final_alarm and not geometry_alarm:
        return "fusion_raised"
    if geometry_alarm and not final_alarm:
        return "fusion_lowered"
    return "aligned_high" if final_alarm else "aligned_low"


def _alarm_stats(
    series: Sequence[WindowPrediction], threshold: float
) -> dict[str, float | int]:
    alarm_count = sum(int(row.score >= threshold) for row in series)
    longest_count = 0
    longest_duration = 0.0
    run_start = 0.0
    run_count = 0
    previous_time = 0.0
    steps = [
        current.end_time_s - previous.end_time_s
        for previous, current in zip(series, series[1:], strict=False)
        if current.end_time_s > previous.end_time_s
    ]
    nominal_step = statistics.median(steps) if steps else 0.0
    for row in series:
        if row.score >= threshold:
            if run_count == 0:
                run_start = row.end_time_s
            run_count += 1
            previous_time = row.end_time_s
            longest_count = max(longest_count, run_count)
            longest_duration = max(
                longest_duration,
                previous_time - run_start + nominal_step,
            )
        else:
            run_count = 0
    return {
        "windows": alarm_count,
        "fraction": alarm_count / len(series) if series else 0.0,
        "longest_run_windows": longest_count,
        "longest_run_s": longest_duration,
    }


def _metrics(
    manifest: dict[str, ManifestVideo],
    predictions: dict[str, VideoPrediction],
    threshold: float,
) -> dict[str, Any]:
    records = [
        VideoRecord(
            domain=record.domain,
            path=path,
            label=record.label,
            score=predictions[path].peak.score if path in predictions else 0.0,
        )
        for path, record in manifest.items()
    ]
    return asdict(classification_metrics(records, threshold))


def _missing_prediction() -> VideoPrediction:
    peak = WindowPrediction(
        window_index=-1,
        end_time_s=0.0,
        score=0.0,
        geometry_score=0.0,
        global_score=0.0,
        roi_score=0.0,
        stage_label=0,
        action_label=9,
        global_attention=0.0,
        roi_attention=0.0,
    )
    return VideoPrediction(
        peak=peak,
        geometry_video_score=0.0,
        global_video_score=0.0,
        roi_video_score=0.0,
        series=(peak,),
    )


def _counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _persistence_summary(
    rows: Sequence[dict[str, Any]], prefix: str
) -> dict[str, float | int]:
    if not rows:
        return {
            "videos": 0,
            "single_window_videos": 0,
            "median_alarm_fraction": 0.0,
            "median_longest_run_windows": 0.0,
            "median_longest_run_s": 0.0,
        }
    return {
        "videos": len(rows),
        "single_window_videos": sum(
            int(row[f"{prefix}_longest_run_windows"] <= 1) for row in rows
        ),
        "median_alarm_fraction": statistics.median(
            float(row[f"{prefix}_alarm_fraction"]) for row in rows
        ),
        "median_longest_run_windows": statistics.median(
            float(row[f"{prefix}_longest_run_windows"]) for row in rows
        ),
        "median_longest_run_s": statistics.median(
            float(row[f"{prefix}_longest_run_s"]) for row in rows
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:160]


def _read_video_frames(path: Path, times: Sequence[float]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for time_s in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                frame = np.full((180, 320, 3), 28, dtype=np.uint8)
                cv2.putText(
                    frame,
                    "frame unavailable",
                    (55, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _load_actor_boxes(path: Path | None) -> list[tuple[float, tuple[float, ...], int]]:
    if path is None or not path.is_file():
        return []
    rows: list[tuple[float, tuple[float, ...], int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            raw_box = raw.get("actor_bbox_xyxy_norm")
            if raw_box is None:
                continue
            box = tuple(float(value) for value in raw_box)
            rows.append(
                (
                    float(raw["timestamp_s"]),
                    box,
                    int(raw.get("actor_source_track_id") or -1),
                )
            )
    return rows


def _overlay_actor_box(
    frame: np.ndarray,
    *,
    time_s: float,
    actor_boxes: Sequence[tuple[float, tuple[float, ...], int]],
) -> np.ndarray:
    if not actor_boxes:
        return frame
    _, box, track_id = min(actor_boxes, key=lambda row: abs(row[0] - time_s))
    output = frame.copy()
    height, width = output.shape[:2]
    x1, y1, x2, y2 = box
    start = (
        min(max(round(x1 * width), 0), width - 1),
        min(max(round(y1 * height), 0), height - 1),
    )
    end = (
        min(max(round(x2 * width), 0), width - 1),
        min(max(round(y2 * height), 0), height - 1),
    )
    cv2.rectangle(output, start, end, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(
        output,
        f"actor track={track_id}",
        (start[0] + 4, max(18, start[1] + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _draw_series(
    baseline: VideoPrediction,
    candidate: VideoPrediction,
    threshold: float,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 55, 18, 20, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_times = [row.end_time_s for row in baseline.series + candidate.series]
    maximum_time = max(all_times, default=1.0)
    maximum_time = max(maximum_time, 1.0)

    def point(time_s: float, score: float) -> tuple[int, int]:
        x = left + round(time_s / maximum_time * plot_width)
        y = top + round((1.0 - min(max(score, 0.0), 1.0)) * plot_height)
        return x, y

    for value in (0.0, 0.5, 1.0):
        y = point(0.0, value)[1]
        cv2.line(canvas, (left, y), (width - right, y), (220, 220, 220), 1)
        cv2.putText(
            canvas,
            f"{value:.1f}",
            (8, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
    threshold_y = point(0.0, threshold)[1]
    cv2.line(
        canvas,
        (left, threshold_y),
        (width - right, threshold_y),
        (80, 80, 210),
        1,
        cv2.LINE_AA,
    )

    curves = [
        ("V3.5", baseline.series, "score", (120, 120, 120)),
        ("V3.6", candidate.series, "score", (30, 30, 210)),
        ("geometry", candidate.series, "geometry_score", (55, 150, 55)),
        ("global", candidate.series, "global_score", (205, 135, 40)),
        ("roi", candidate.series, "roi_score", (180, 55, 160)),
    ]
    legend_x = left
    for label, series, attribute, color in curves:
        points = np.asarray(
            [point(row.end_time_s, float(getattr(row, attribute))) for row in series],
            dtype=np.int32,
        )
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        cv2.line(canvas, (legend_x, height - 17), (legend_x + 18, height - 17), color, 2)
        cv2.putText(
            canvas,
            label,
            (legend_x + 23, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        legend_x += 110
    cv2.putText(
        canvas,
        f"time 0..{maximum_time:.1f}s; locked threshold={threshold:.4f}",
        (width - 355, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_review_panel(
    *,
    record: ManifestVideo,
    baseline: VideoPrediction,
    candidate: VideoPrediction,
    threshold: float,
    reason: str,
    output_path: Path,
    actor_targets_path: Path | None = None,
) -> bool:
    if record.video_path is None or not record.video_path.is_file():
        return False
    peak_time = candidate.peak.end_time_s
    geometry_peak = max(
        candidate.series,
        key=lambda row: (row.geometry_score, -row.window_index),
    ).end_time_s
    end_time = candidate.series[-1].end_time_s
    times = sorted({0.0, peak_time, geometry_peak, end_time})
    for fraction in (0.25, 0.5, 0.75):
        if len(times) >= 4:
            break
        times.append(end_time * fraction)
        times = sorted(set(times))
    times = times[:4]
    raw_frames = _read_video_frames(record.video_path, times)
    actor_boxes = _load_actor_boxes(actor_targets_path)
    tiles: list[np.ndarray] = []
    for frame, time_s in zip(raw_frames, times, strict=True):
        frame = _overlay_actor_box(
            frame,
            time_s=time_s,
            actor_boxes=actor_boxes,
        )
        tile = _letterbox(frame, 320, 180)
        cv2.rectangle(tile, (0, 0), (100, 23), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"t={time_s:.2f}s",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    frames = np.concatenate(tiles, axis=1)
    header = np.full((78, frames.shape[1], 3), 245, dtype=np.uint8)
    lines = [
        f"{record.domain} | label={int(record.label)} | {reason}",
        f"{record.path}",
        (
            f"V3.5={baseline.peak.score:.4f}  V3.6={candidate.peak.score:.4f}  "
            f"geo={candidate.geometry_video_score:.4f}  global={candidate.global_video_score:.4f}  "
            f"roi={candidate.roi_video_score:.4f}"
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line[:180],
            (10, 20 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    plot = _draw_series(
        baseline,
        candidate,
        threshold,
        width=frames.shape[1],
        height=260,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), np.concatenate([header, frames, plot])))


def _review_rows(
    rows: Sequence[dict[str, Any]],
    *,
    per_category: int,
) -> list[dict[str, Any]]:
    reasons = {
        "p90_new_fp": lambda row: row["p90_transition"] == "new_fp",
        "p90_common_fp": lambda row: row["p90_transition"] == "common_fp",
        "locked_fp_fusion_raised": lambda row: row["candidate_error"] == "fp"
        and row["fusion_crossing"] == "fusion_raised",
        "locked_fp_geometry_high": lambda row: row["candidate_error"] == "fp"
        and row["fusion_crossing"] == "aligned_high",
        "locked_fn_fusion_lowered": lambda row: row["candidate_error"] == "fn"
        and row["fusion_crossing"] == "fusion_lowered",
        "locked_fn_geometry_low": lambda row: row["candidate_error"] == "fn"
        and row["fusion_crossing"] == "aligned_low",
        "baseline_only_correct": lambda row: row["locked_transition"]
        == "baseline_only_correct",
        "candidate_only_correct": lambda row: row["locked_transition"]
        == "candidate_only_correct",
    }
    selected: list[dict[str, Any]] = []
    domains = sorted({str(row["domain"]) for row in rows})
    for domain in domains:
        domain_rows = [row for row in rows if row["domain"] == domain]
        for reason, predicate in reasons.items():
            candidates = [row for row in domain_rows if predicate(row)]
            reverse = "fn" not in reason
            candidates.sort(
                key=lambda row: (
                    float(row["candidate_score"]),
                    abs(float(row["score_delta"])),
                ),
                reverse=reverse,
            )
            for row in candidates[:per_category]:
                selected.append({"review_reason": reason, **row})
    return selected


def audit_online(
    *,
    domain: str,
    online_videos_path: Path,
    online_frames_path: Path,
    manifest: dict[str, ManifestVideo],
    offline_predictions: dict[str, VideoPrediction],
    threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    frame_rows: dict[str, list[dict[str, str]]] = {}
    with online_frames_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_rows.setdefault(str(row["path"]), []).append(row)

    comparisons: list[dict[str, Any]] = []
    with online_videos_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = str(row["path"])
            if path not in manifest or path not in offline_predictions:
                raise ValueError(f"Online path is absent from offline audit: {path}")
            record = manifest[path]
            online_score = _float(row, "video_score")
            offline_score = offline_predictions[path].peak.score
            online_error = _classification(
                record.label, online_score >= threshold
            )
            offline_error = _classification(
                record.label, offline_score >= threshold
            )
            frames = sorted(
                frame_rows.get(path, []),
                key=lambda frame: _int(frame, "sample_index"),
            )
            valid_track_ids = [
                _int(frame, "actor_track_id", -1)
                for frame in frames
                if _int(frame, "actor_track_id", -1) >= 0
            ]
            track_switches = sum(
                current != previous
                for previous, current in zip(
                    valid_track_ids, valid_track_ids[1:], strict=False
                )
            )
            missing_actor_frames = sum(
                not str(frame.get("actor_box", "")).strip() for frame in frames
            )
            peak_frame = max(
                frames,
                key=lambda frame: _float(frame, "score"),
                default=None,
            )
            comparisons.append(
                {
                    "domain": domain,
                    "path": path,
                    "label": int(record.label),
                    "online_score": online_score,
                    "offline_score": offline_score,
                    "online_minus_offline": online_score - offline_score,
                    "online_error": online_error,
                    "offline_error": offline_error,
                    "correctness_transition": _transition(
                        offline_error, online_error
                    ),
                    "sampled_frames": len(frames),
                    "track_switches": track_switches,
                    "unique_track_ids": len(set(valid_track_ids)),
                    "missing_actor_frames": missing_actor_frames,
                    "online_peak_time_s": (
                        _float(peak_frame, "timestamp_s") if peak_frame else 0.0
                    ),
                    "online_peak_geometry_score": (
                        _float(peak_frame, "geometry_score") if peak_frame else 0.0
                    ),
                    "online_peak_roi_attention": (
                        _float(peak_frame, "roi_attention") if peak_frame else 0.0
                    ),
                }
            )

    online_records = [
        VideoRecord(
            domain=domain,
            path=str(row["path"]),
            label=bool(row["label"]),
            score=float(row["online_score"]),
        )
        for row in comparisons
    ]
    offline_records = [
        VideoRecord(
            domain=domain,
            path=str(row["path"]),
            label=bool(row["label"]),
            score=float(row["offline_score"]),
        )
        for row in comparisons
    ]
    non_agreements = [
        row
        for row in comparisons
        if row["correctness_transition"] != "both_correct"
    ]
    _write_csv(output_dir / "online_comparison.csv", comparisons)
    _write_csv(output_dir / "online_review_manifest.csv", non_agreements)
    return {
        "videos": len(comparisons),
        "online_metrics": asdict(
            classification_metrics(online_records, threshold)
        ),
        "offline_metrics": asdict(
            classification_metrics(offline_records, threshold)
        ),
        "correctness_transitions": _counts(
            comparisons, "correctness_transition"
        ),
        "online_errors": _counts(comparisons, "online_error"),
        "non_agreement_tracking": {
            "videos": len(non_agreements),
            "videos_with_track_switches": sum(
                int(row["track_switches"] > 0) for row in non_agreements
            ),
            "videos_with_missing_actor_frames": sum(
                int(row["missing_actor_frames"] > 0) for row in non_agreements
            ),
        },
        "inputs": {
            "videos": str(online_videos_path),
            "frames": str(online_frames_path),
        },
    }


def audit(
    *,
    manifest_path: Path,
    split: str,
    baseline_predictions: Sequence[tuple[str, Path]],
    candidate_predictions: Sequence[tuple[str, Path]],
    threshold: float,
    output_dir: Path,
    review_per_category: int,
    render_panels: bool,
    video_roots: Sequence[tuple[str, Path]] = (),
    actor_target_roots: Sequence[tuple[str, Path]] = (),
    online_domain: str | None = None,
    online_videos_path: Path | None = None,
    online_frames_path: Path | None = None,
) -> dict[str, Any]:
    baseline_paths = dict(baseline_predictions)
    candidate_paths = dict(candidate_predictions)
    if baseline_paths.keys() != candidate_paths.keys():
        raise ValueError("Baseline and candidate domains must match")
    domains = set(baseline_paths)
    if not domains:
        raise ValueError("At least one prediction domain is required")
    manifest = load_manifest(
        manifest_path,
        split=split,
        domains=domains,
        video_roots=dict(video_roots),
    )

    baseline_by_domain: dict[str, dict[str, VideoPrediction]] = {}
    candidate_by_domain: dict[str, dict[str, VideoPrediction]] = {}
    missing: dict[str, dict[str, int]] = {}
    summary_domains: dict[str, Any] = {}
    failure_rows: list[dict[str, Any]] = []

    for domain in sorted(domains):
        expected = set(manifest[domain])
        baseline, baseline_missing = load_predictions(
            baseline_paths[domain], expected_paths=expected
        )
        candidate, candidate_missing = load_predictions(
            candidate_paths[domain], expected_paths=expected
        )
        baseline_by_domain[domain] = baseline
        candidate_by_domain[domain] = candidate
        missing[domain] = {
            "baseline": baseline_missing,
            "candidate": candidate_missing,
        }
        points = {
            "baseline_p90": precision_at_recall(baseline, manifest[domain], 0.90),
            "baseline_p95": precision_at_recall(baseline, manifest[domain], 0.95),
            "candidate_p90": precision_at_recall(candidate, manifest[domain], 0.90),
            "candidate_p95": precision_at_recall(candidate, manifest[domain], 0.95),
        }
        persistence_points: dict[str, Any] = {}
        for run_length in (1, 2, 3):
            persistence_scores = {
                path: _persistent_score(candidate[path], run_length)
                if path in candidate
                else 0.0
                for path in manifest[domain]
            }
            persistence_points[str(run_length)] = {
                "p90": precision_at_recall_scores(
                    persistence_scores, manifest[domain], 0.90
                ),
                "p95": precision_at_recall_scores(
                    persistence_scores, manifest[domain], 0.95
                ),
            }

        for path, record in manifest[domain].items():
            baseline_prediction = baseline.get(path)
            candidate_prediction = candidate.get(path)
            baseline_is_missing = baseline_prediction is None
            candidate_is_missing = candidate_prediction is None
            baseline_prediction = baseline_prediction or _missing_prediction()
            candidate_prediction = candidate_prediction or _missing_prediction()
            baseline_score = baseline_prediction.peak.score
            candidate_score = candidate_prediction.peak.score
            baseline_alarm = baseline_score >= threshold
            candidate_alarm = candidate_score >= threshold
            geometry_alarm = candidate_prediction.geometry_video_score >= threshold
            baseline_error = _classification(record.label, baseline_alarm)
            candidate_error = _classification(record.label, candidate_alarm)
            p90_transition = _operating_transition(
                label=record.label,
                baseline_alarm=baseline_score
                >= float(points["baseline_p90"]["threshold"]),
                candidate_alarm=candidate_score
                >= float(points["candidate_p90"]["threshold"]),
            )
            p95_transition = _operating_transition(
                label=record.label,
                baseline_alarm=baseline_score
                >= float(points["baseline_p95"]["threshold"]),
                candidate_alarm=candidate_score
                >= float(points["candidate_p95"]["threshold"]),
            )
            peak = candidate_prediction.peak
            locked_alarm = _alarm_stats(candidate_prediction.series, threshold)
            p90_alarm = _alarm_stats(
                candidate_prediction.series,
                float(points["candidate_p90"]["threshold"]),
            )
            failure_rows.append(
                {
                    "domain": domain,
                    "path": path,
                    "label": int(record.label),
                    "label_name": record.label_name,
                    "semantic_group": _semantic_group(
                        domain, path, record.label_name
                    ),
                    "video_path": str(record.video_path or ""),
                    "baseline_missing": int(baseline_is_missing),
                    "candidate_missing": int(candidate_is_missing),
                    "baseline_score": baseline_score,
                    "candidate_score": candidate_score,
                    "score_delta": candidate_score - baseline_score,
                    "geometry_video_score": candidate_prediction.geometry_video_score,
                    "global_video_score": candidate_prediction.global_video_score,
                    "roi_video_score": candidate_prediction.roi_video_score,
                    "peak_time_s": peak.end_time_s,
                    "peak_window_index": peak.window_index,
                    "peak_stage_label": peak.stage_label,
                    "peak_action_label": peak.action_label,
                    "peak_action_name": ACTION_NAMES.get(
                        peak.action_label, str(peak.action_label)
                    ),
                    "peak_geometry_score": peak.geometry_score,
                    "peak_global_score": peak.global_score,
                    "peak_roi_score": peak.roi_score,
                    "peak_global_attention": peak.global_attention,
                    "peak_roi_attention": peak.roi_attention,
                    "baseline_error": baseline_error,
                    "candidate_error": candidate_error,
                    "locked_transition": _transition(
                        baseline_error, candidate_error
                    ),
                    "fusion_crossing": _fusion_crossing(
                        candidate_alarm, geometry_alarm
                    ),
                    "p90_transition": p90_transition,
                    "p95_transition": p95_transition,
                    "locked_alarm_windows": locked_alarm["windows"],
                    "locked_alarm_fraction": locked_alarm["fraction"],
                    "locked_longest_run_windows": locked_alarm[
                        "longest_run_windows"
                    ],
                    "locked_longest_run_s": locked_alarm["longest_run_s"],
                    "p90_alarm_windows": p90_alarm["windows"],
                    "p90_alarm_fraction": p90_alarm["fraction"],
                    "p90_longest_run_windows": p90_alarm[
                        "longest_run_windows"
                    ],
                    "p90_longest_run_s": p90_alarm["longest_run_s"],
                }
            )

        domain_rows = [row for row in failure_rows if row["domain"] == domain]
        candidate_errors = [
            row for row in domain_rows if row["candidate_error"] in {"fp", "fn"}
        ]
        locked_false_positives = [
            row for row in domain_rows if row["candidate_error"] == "fp"
        ]
        p90_false_positives = [
            row
            for row in domain_rows
            if row["p90_transition"] in {"new_fp", "common_fp"}
        ]
        summary_domains[domain] = {
            "videos": len(manifest[domain]),
            "positives": sum(int(row.label) for row in manifest[domain].values()),
            "operating_points": points,
            "candidate_persistence_operating_points": persistence_points,
            "locked_threshold": {
                "baseline": _metrics(manifest[domain], baseline, threshold),
                "candidate": _metrics(manifest[domain], candidate, threshold),
                "candidate_errors": _counts(domain_rows, "candidate_error"),
                "transitions": _counts(domain_rows, "locked_transition"),
                "fusion_crossings_among_errors": _counts(
                    candidate_errors, "fusion_crossing"
                ),
                "errors_by_semantic_group": _counts(
                    candidate_errors, "semantic_group"
                ),
                "errors_by_peak_action": _counts(
                    candidate_errors, "peak_action_name"
                ),
                "false_positive_persistence": _persistence_summary(
                    locked_false_positives, "locked"
                ),
            },
            "p90_transitions": _counts(domain_rows, "p90_transition"),
            "p90_false_positive_persistence": _persistence_summary(
                p90_false_positives, "p90"
            ),
            "p95_transitions": _counts(domain_rows, "p95_transition"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = _review_rows(
        failure_rows, per_category=review_per_category
    )
    rendered: dict[tuple[str, str], str] = {}
    actor_target_roots_by_domain = dict(actor_target_roots)
    if render_panels:
        for row in review_rows:
            key = (str(row["domain"]), str(row["path"]))
            if key not in rendered:
                domain, path = key
                output_path = (
                    output_dir
                    / "review_panels"
                    / domain
                    / f"{_safe_name(path)}.jpg"
                )
                ok = render_review_panel(
                    record=manifest[domain][path],
                    baseline=baseline_by_domain[domain][path],
                    candidate=candidate_by_domain[domain][path],
                    threshold=threshold,
                    reason=str(row["review_reason"]),
                    output_path=output_path,
                    actor_targets_path=(
                        actor_target_roots_by_domain[domain]
                        / Path(path.replace("\\", "/")).with_suffix(".jsonl")
                        if domain in actor_target_roots_by_domain
                        else None
                    ),
                )
                rendered[key] = str(output_path) if ok else ""
            row["review_panel"] = rendered[key]
    else:
        for row in review_rows:
            key = (str(row["domain"]), str(row["path"]))
            output_path = (
                output_dir
                / "review_panels"
                / key[0]
                / f"{_safe_name(key[1])}.jpg"
            )
            rendered[key] = str(output_path) if output_path.is_file() else ""
            row["review_panel"] = rendered[key]

    _write_csv(output_dir / "failure_cases.csv", failure_rows)
    _write_csv(output_dir / "review_manifest.csv", review_rows)
    online_result = None
    if online_domain or online_videos_path or online_frames_path:
        if not online_domain or online_videos_path is None or online_frames_path is None:
            raise ValueError(
                "Online audit requires domain, videos CSV, and frames CSV"
            )
        if online_domain not in candidate_by_domain:
            raise ValueError(f"Unknown online domain: {online_domain}")
        online_result = audit_online(
            domain=online_domain,
            online_videos_path=online_videos_path,
            online_frames_path=online_frames_path,
            manifest=manifest[online_domain],
            offline_predictions=candidate_by_domain[online_domain],
            threshold=threshold,
            output_dir=output_dir,
        )

    result = {
        "protocol": {
            "split": split,
            "locked_threshold": threshold,
            "threshold_source": "validation_only",
            "test_usage": "diagnostic_only",
        },
        "inputs": {
            "manifest": str(manifest_path),
            "baseline_predictions": {
                domain: str(path) for domain, path in baseline_predictions
            },
            "candidate_predictions": {
                domain: str(path) for domain, path in candidate_predictions
            },
            "video_roots": {
                domain: str(path) for domain, path in video_roots
            },
            "actor_target_roots": {
                domain: str(path) for domain, path in actor_target_roots
            },
        },
        "missing_videos": missing,
        "domains": summary_domains,
        "review": {
            "per_category": review_per_category,
            "rows": len(review_rows),
            "unique_panels": sum(bool(path) for path in rendered.values()),
        },
        "online": online_result,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--baseline-predictions",
        type=_mapping,
        action="append",
        required=True,
        metavar="DOMAIN=CSV",
    )
    parser.add_argument(
        "--candidate-predictions",
        type=_mapping,
        action="append",
        required=True,
        metavar="DOMAIN=CSV",
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video-root",
        type=_mapping,
        action="append",
        default=[],
        metavar="DOMAIN=DIR",
    )
    parser.add_argument(
        "--actor-target-root",
        type=_mapping,
        action="append",
        default=[],
        metavar="DOMAIN=DIR",
    )
    parser.add_argument("--review-per-category", type=int, default=6)
    parser.add_argument("--render-panels", action="store_true")
    parser.add_argument("--online-domain")
    parser.add_argument("--online-videos", type=Path)
    parser.add_argument("--online-frames", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit(
        manifest_path=args.manifest,
        split=args.split,
        baseline_predictions=args.baseline_predictions,
        candidate_predictions=args.candidate_predictions,
        threshold=args.threshold,
        output_dir=args.output_dir,
        review_per_category=args.review_per_category,
        render_panels=args.render_panels,
        video_roots=args.video_root,
        actor_target_roots=args.actor_target_root,
        online_domain=args.online_domain,
        online_videos_path=args.online_videos,
        online_frames_path=args.online_frames,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
