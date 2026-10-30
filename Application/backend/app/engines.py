"""Real-time and recorded-result inference engines."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import CHECKPOINT, DEVICE, FINAL_ROOT, INFERENCE_ROOT, YOLO_WEIGHTS


class InferenceEngine(Protocol):
    mode: str
    label: str
    device: str
    effective_fps: float
    fallback_reason: str | None

    def reset(self, clip_name: str) -> None: ...

    def predict(
        self, frame: np.ndarray, sample_index: int, clip_name: str
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _normalise_prediction(row: dict[str, Any]) -> dict[str, Any]:
    people = row.get("people", [])
    if isinstance(people, str):
        people = json.loads(people)
    actor_box = row.get("actor_box")
    if isinstance(actor_box, str):
        actor_box = json.loads(actor_box) if actor_box else None
    return {
        "sample_index": int(row.get("sample_index", 0)),
        "timestamp_s": float(row.get("timestamp_s", 0.0)),
        "score": float(row.get("score", 0.0)),
        "geometry_score": float(row.get("geometry_score", 0.0)),
        "global_attention": _optional_float(row.get("global_attention")),
        "roi_attention": _optional_float(row.get("roi_attention")),
        "actor_track_id": _optional_int(row.get("actor_track_id")),
        "actor_box": actor_box,
        "people": people,
        "detector_ms": float(row.get("detector_ms", 0.0)),
        "tracking_ms": float(row.get("tracking_ms", 0.0)),
        "feature_pool_ms": float(row.get("feature_pool_ms", 0.0)),
        "confirmer_ms": float(row.get("confirmer_ms", 0.0)),
        "algorithm_ms": float(row.get("algorithm_ms", 0.0)),
    }


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


class ReplayEngine:
    """Play the packaged frame predictions in sync with the original clips."""

    mode = "replay"
    label = "结果回放"
    device = "已记录的 CUDA 推理（非实时）"
    effective_fps = 10.0

    def __init__(self, fallback_reason: str | None = None) -> None:
        self.fallback_reason = fallback_reason
        self._rows: dict[str, list[dict[str, Any]]] = {}
        frames_path = INFERENCE_ROOT / "frames.csv"
        with frames_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = Path(row["path"]).name
                self._rows.setdefault(name, []).append(_normalise_prediction(row))
        for rows in self._rows.values():
            rows.sort(key=lambda item: item["sample_index"])

    def reset(self, clip_name: str) -> None:
        if clip_name not in self._rows:
            raise KeyError(f"No recorded predictions for {clip_name}")

    def predict(
        self, frame: np.ndarray, sample_index: int, clip_name: str
    ) -> dict[str, Any]:
        del frame
        rows = self._rows[clip_name]
        return dict(rows[min(sample_index, len(rows) - 1)])

    def close(self) -> None:
        return None


class LiveEngine:
    """Run the packaged HN3-B-R model for every sampled stream frame."""

    mode = "live"
    label = "实时模型"
    fallback_reason = None

    def __init__(self) -> None:
        if str(FINAL_ROOT) not in sys.path:
            sys.path.insert(0, str(FINAL_ROOT))
        import torch

        from edgefall_st.runtime.fygr_online import OnlineFygrRuntime

        if DEVICE == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(DEVICE)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EDGEFALL_DEVICE requests CUDA, but CUDA is unavailable")
        self.device = (
            torch.cuda.get_device_name(torch_device)
            if torch_device.type == "cuda"
            else "CPU"
        )
        self._runtime = OnlineFygrRuntime(
            checkpoint_path=CHECKPOINT,
            weights_path=YOLO_WEIGHTS,
            device=torch_device,
            amp=True,
            confidence=0.10,
            iou=0.30,
            tracker="bytetrack.yaml",
            quantize=16,
            max_actor_missed=2,
            max_people=2,
        )
        self.effective_fps = float(self._runtime.effective_fps)

    def reset(self, clip_name: str) -> None:
        del clip_name
        self._runtime.reset_stream()

    def predict(
        self, frame: np.ndarray, sample_index: int, clip_name: str
    ) -> dict[str, Any]:
        del clip_name
        return _normalise_prediction(asdict(
            self._runtime.process_frame(frame, sample_index=sample_index)
        ))

    def close(self) -> None:
        self._runtime.close()


def build_engine(mode: str) -> InferenceEngine:
    if mode == "replay":
        return ReplayEngine()
    if mode not in {"auto", "live"}:
        raise ValueError("EDGEFALL_ENGINE must be auto, live, or replay")
    try:
        return LiveEngine()
    except Exception as error:
        if mode == "live":
            raise
        reason = f"{type(error).__name__}: {error}"
        return ReplayEngine(fallback_reason=reason)
