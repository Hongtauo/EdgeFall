"""Cache-free online YOLO + FYGR fall-event inference and benchmarking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor

from edgefall_st.data.dataset import (
    _letterbox_parameters,
    _transform_normalized_box,
)
from edgefall_st.features.adapter import UltralyticsYOLOFeatureAdapter
from edgefall_st.features.cache import roi_feature_shape
from edgefall_st.models.fygr import (
    FYGR_FUSION_CONCAT_MLP,
    FrozenYoloGlobalROIConfirmer,
)
from edgefall_st.pseudo.batched_tracker import _build_bytetracker
from edgefall_st.pseudo.yolo26_tracker import TeacherConfig


@dataclass(frozen=True)
class TrackedPerson:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None


@dataclass(frozen=True)
class OnlinePersonPrediction:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    score: float
    falling_score: float
    fallen_score: float
    geometry_score: float


@dataclass(frozen=True)
class OnlinePrediction:
    sample_index: int
    timestamp_s: float
    score: float
    geometry_score: float
    global_attention: float | None
    roi_attention: float | None
    actor_track_id: int | None
    actor_box: tuple[float, float, float, float] | None
    people: tuple[OnlinePersonPrediction, ...]
    detector_ms: float
    tracking_ms: float
    feature_pool_ms: float
    confirmer_ms: float
    algorithm_ms: float
    timed: bool = True


@dataclass(frozen=True)
class VideoOnlineResult:
    path: str
    label: int | None
    source_fps: float
    source_frames: int
    sampled_frames: int
    video_score: float
    first_alarm_s: float | None
    decode_ms: float
    wall_s: float


def _attention_masses(
    attention_weights: Tensor,
    *,
    use_global: bool,
    use_roi: bool,
    visual_slots: int,
) -> tuple[float | None, float | None]:
    """Summarize current-query attention for online diagnostics."""

    if attention_weights.ndim == 2:
        current = attention_weights[0]
        slots_per_modality = 1
    elif attention_weights.ndim == 3:
        current = attention_weights[0, -1]
        slots_per_modality = int(visual_slots)
    else:
        raise ValueError("Unexpected fusion attention shape")
    expected = slots_per_modality * (int(use_global) + int(use_roi))
    if current.numel() != expected:
        raise ValueError("Fusion attention does not match enabled modalities")

    index = 0
    global_attention: float | None = None
    roi_attention: float | None = None
    if use_global:
        global_attention = float(
            current[index : index + slots_per_modality].float().sum().item()
        )
        index += slots_per_modality
    if use_roi:
        roi_attention = float(
            current[index : index + slots_per_modality].float().sum().item()
        )
    return global_attention, roi_attention


class CausalActorSelector:
    """Select one actor using only current and previous tracking state."""

    def __init__(self, *, max_missed: int = 2) -> None:
        if max_missed < 0:
            raise ValueError("max_missed cannot be negative")
        self.max_missed = int(max_missed)
        self.active_track_id: int | None = None
        self.missed = 0

    @staticmethod
    def _priority(person: TrackedPerson, frame_shape: Sequence[int]) -> float:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        left, top, right, bottom = person.bbox_xyxy
        area = max(0.0, right - left) * max(0.0, bottom - top)
        area_ratio = area / max(1.0, float(height * width))
        center_x = (left + right) * 0.5 / max(1.0, width)
        center_y = (top + bottom) * 0.5 / max(1.0, height)
        center_distance = math.hypot(center_x - 0.5, center_y - 0.5)
        return person.confidence + math.sqrt(area_ratio) - 0.1 * center_distance

    def select(
        self,
        people: Sequence[TrackedPerson],
        frame_shape: Sequence[int],
    ) -> TrackedPerson | None:
        if self.active_track_id is not None:
            for person in people:
                if person.track_id == self.active_track_id:
                    self.missed = 0
                    return person
            self.missed += 1
            if self.missed <= self.max_missed:
                return None

        if not people:
            return None
        selected = max(
            people,
            key=lambda person: self._priority(person, frame_shape),
        )
        self.active_track_id = selected.track_id
        self.missed = 0
        return selected


@dataclass
class _OnlinePersonFeature:
    roi_feature: Tensor
    geometry_box: Tensor
    actor_valid: Tensor


@dataclass
class _OnlineFrameFeature:
    global_feature: Tensor
    people: dict[int, _OnlinePersonFeature]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))


class OnlineFygrRuntime:
    """Run detector, tracker, feature pooling, and confirmer without a cache."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        weights_path: Path,
        device: torch.device,
        amp: bool = True,
        confidence: float = 0.10,
        iou: float = 0.30,
        tracker: str = "bytetrack.yaml",
        quantize: int | None = 16,
        max_actor_missed: int = 2,
        max_people: int = 2,
    ) -> None:
        if max_people <= 0:
            raise ValueError("max_people must be positive")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("version") not in {
            "v3.4-fygr",
            "v3.5-md-fygr",
            "v3.6-gcva",
            "v3.6-md-gcva",
            "v3.7-person-mil",
            "v3.7-md-person-mil",
            "v3.8-temporal-token",
            "v3.8-md-temporal-token",
            "v3.8-person-mil-temporal-token",
            "v3.8-md-person-mil-temporal-token",
            "v3.9-summary-ranking",
            "v3.9-md-summary-ranking",
            "v3.10-hard-tail-ranking",
            "v3.10-md-hard-tail-ranking",
        }:
            raise ValueError("Checkpoint is not a supported FYGR checkpoint")
        contract = checkpoint["feature_contract"]
        if _sha256(weights_path) != contract["weights_sha256"]:
            raise ValueError("YOLO weights do not match the checkpoint contract")
        saved = argparse.Namespace(**checkpoint["arguments"])
        geometry_checkpoint = Path(saved.geometry_checkpoint)
        if not geometry_checkpoint.is_absolute():
            geometry_checkpoint = checkpoint_path.parent / geometry_checkpoint
        if not geometry_checkpoint.is_file():
            raise FileNotFoundError(
                f"Geometry checkpoint not found: {geometry_checkpoint}"
            )
        adapter_spec = contract["feature_adapter"]
        self.adapter = UltralyticsYOLOFeatureAdapter(
            weights_path,
            global_levels=tuple(adapter_spec["global_levels"]),
            roi_levels=tuple(adapter_spec["roi_levels"]),
            roi_output_size=int(adapter_spec["roi_output_size"]),
            roi_context_scale=float(adapter_spec["roi_context_scale"]),
        ).to(device)
        self.adapter.eval()
        arrays = contract["arrays"]
        self.model = FrozenYoloGlobalROIConfirmer(
            geometry_checkpoint=geometry_checkpoint,
            history_offsets=tuple(contract["history_offsets"]),
            visual_offsets=tuple(contract["visual_offsets"]),
            global_feature_dim=int(arrays["global_features"][-1]),
            roi_feature_shape=roi_feature_shape(contract),
            temporal_dim=int(saved.temporal_dim),
            dropout=float(saved.dropout),
            use_global=bool(saved.use_global),
            use_roi=bool(saved.use_roi),
            fusion_mode=str(
                getattr(saved, "fusion_mode", FYGR_FUSION_CONCAT_MLP)
            ),
            attention_heads=int(getattr(saved, "attention_heads", 4)),
            attention_ffn_dim=int(
                getattr(saved, "attention_ffn_dim", 256)
            ),
            person_mil_temperature=float(
                getattr(saved, "person_mil_temperature", 0.0)
            ),
        ).to(device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self._detector_parameter_count = sum(
            parameter.numel() for parameter in self.adapter.detector.parameters()
        )

        self.checkpoint_path = checkpoint_path
        self.weights_path = weights_path
        self.device = device
        self.contract = contract
        self.history_offsets = tuple(
            int(value) for value in contract["history_offsets"]
        )
        self.visual_offsets = tuple(
            int(value) for value in contract["visual_offsets"]
        )
        self.visual_input_size = tuple(
            int(value) for value in contract["visual_input_size"]
        )
        self.geometry_input_size = tuple(
            int(value) for value in contract["geometry_input_size"]
        )
        self.effective_fps = float(contract["effective_fps"])
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.tracker_path = str(tracker)
        self.quantize = quantize
        self.max_actor_missed = int(max_actor_missed)
        self.max_people = int(max_people)
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if self.amp_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self._history: deque[_OnlineFrameFeature] = deque(
            maxlen=max(self.history_offsets) + 1
        )
        self._tracker: Any = None
        self._selector: CausalActorSelector | None = None
        self.reset_stream()

    @property
    def detector_parameter_count(self) -> int:
        return self._detector_parameter_count

    @property
    def fused_detector_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.adapter.detector.parameters()
        )

    @property
    def confirmer_parameter_count(self) -> int:
        return self.model.parameter_count

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_stream(self) -> None:
        config = TeacherConfig(
            model=str(self.weights_path),
            image_size=self.visual_input_size,
            confidence=self.confidence,
            iou=self.iou,
            tracker=self.tracker_path,
            device="0" if self.device.type == "cuda" else "cpu",
            quantize=self.quantize,
            end2end=False,
        )
        self._tracker = _build_bytetracker(config)
        self._selector = CausalActorSelector(max_missed=self.max_actor_missed)
        self._history.clear()

    @staticmethod
    def _tracked_people(
        result: Any,
        frame: np.ndarray,
        tracker: Any,
    ) -> list[TrackedPerson]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if boxes is not None:
                tracker.update(boxes.cpu().numpy(), frame)
            return []
        detections = boxes.cpu().numpy()
        tracks = tracker.update(detections, frame)
        if len(tracks) > 0:
            return [
                TrackedPerson(
                    bbox_xyxy=tuple(float(value) for value in track[:4]),
                    confidence=float(track[5]),
                    track_id=int(track[4]),
                )
                for track in tracks
            ]
        return [
            TrackedPerson(
                bbox_xyxy=tuple(float(value) for value in xyxy),
                confidence=float(confidence),
                track_id=None,
            )
            for xyxy, confidence in zip(
                np.asarray(detections.xyxy).tolist(),
                np.asarray(detections.conf).tolist(),
                strict=True,
            )
        ]

    @staticmethod
    def _normalized_source_box(
        actor: TrackedPerson | None,
        frame: np.ndarray,
    ) -> tuple[float, float, float, float] | None:
        if actor is None:
            return None
        height, width = frame.shape[:2]
        left, top, right, bottom = actor.bbox_xyxy
        return (
            left / width,
            top / height,
            right / width,
            bottom / height,
        )

    @staticmethod
    def _letterboxed_box(
        source_box: tuple[float, float, float, float] | None,
        source_size: tuple[int, int],
        input_size: tuple[int, int],
    ) -> tuple[float, float, float, float] | None:
        scale, pad_left, pad_top = _letterbox_parameters(source_size, input_size)
        transformed = _transform_normalized_box(
            None if source_box is None else list(source_box),
            source_size=source_size,
            input_size=input_size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        return (
            None
            if transformed is None
            else tuple(float(value) for value in transformed)
        )

    def _history_inputs(
        self,
        person_keys: Sequence[int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        latest = self._history[-1]

        def value_at(offset: int) -> _OnlineFrameFeature | None:
            return self._history[-1 - offset] if offset < len(self._history) else None

        history = [value_at(offset) for offset in self.history_offsets]
        visual = [value_at(offset) for offset in self.visual_offsets]
        reference = next(iter(latest.people.values()))
        actor_boxes = torch.stack(
            [
                torch.stack(
                    [
                        reference.geometry_box.new_zeros(4)
                        if item is None or key not in item.people
                        else item.people[key].geometry_box
                        for key in person_keys
                    ]
                )
                for item in history
            ]
        ).unsqueeze(0)
        valid_mask = reference.actor_valid.new_tensor(
            [[float(item is not None) for item in history]]
        )
        actor_valid = torch.stack(
            [
                torch.stack(
                    [
                        reference.actor_valid.new_zeros(())
                        if item is None or key not in item.people
                        else item.people[key].actor_valid
                        for key in person_keys
                    ]
                )
                for item in history
            ]
        ).unsqueeze(0)
        global_features = torch.stack(
            [
                latest.global_feature.new_zeros(latest.global_feature.shape)
                if item is None
                else item.global_feature
                for item in visual
            ]
        ).unsqueeze(0)
        roi_features = torch.stack(
            [
                torch.stack(
                    [
                        reference.roi_feature.new_zeros(
                            reference.roi_feature.shape
                        )
                        if item is None or key not in item.people
                        else item.people[key].roi_feature
                        for key in person_keys
                    ]
                )
                for item in visual
            ]
        ).unsqueeze(0)
        return global_features, roi_features, actor_boxes, valid_mask, actor_valid

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        sample_index: int,
    ) -> OnlinePrediction:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape [H, W, 3]")
        self._sync()
        algorithm_started = time.perf_counter()
        self.adapter.clear_captured_features()
        results = self.adapter.yolo.predict(
            source=[frame_bgr],
            stream=False,
            batch=1,
            classes=[0],
            imgsz=list(self.visual_input_size),
            conf=self.confidence,
            iou=self.iou,
            device=0 if self.device.type == "cuda" else "cpu",
            quantize=self.quantize,
            rect=False,
            verbose=False,
            save=False,
        )
        self._sync()
        detector_done = time.perf_counter()
        if len(results) != 1:
            raise RuntimeError(f"Expected one YOLO result, got {len(results)}")

        people = self._tracked_people(results[0], frame_bgr, self._tracker)
        if self._selector is None:
            raise RuntimeError("Stream selector is not initialized")
        if self.max_people == 1:
            actor = self._selector.select(people, frame_bgr.shape)
            selected_people = [] if actor is None else [actor]
        else:
            selected_people = sorted(
                people,
                key=lambda person: CausalActorSelector._priority(
                    person, frame_bgr.shape
                ),
                reverse=True,
            )[: self.max_people]
        tracking_done = time.perf_counter()
        source_size = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))
        runtime_people: list[
            tuple[
                int,
                TrackedPerson | None,
                tuple[float, float, float, float] | None,
                tuple[float, float, float, float] | None,
            ]
        ] = []
        for person_index, person in enumerate(selected_people):
            source_box = self._normalized_source_box(person, frame_bgr)
            runtime_people.append(
                (
                    person.track_id
                    if person.track_id is not None
                    else -(person_index + 1),
                    person,
                    self._letterboxed_box(
                        source_box, source_size, self.visual_input_size
                    ),
                    self._letterboxed_box(
                        source_box, source_size, self.geometry_input_size
                    ),
                )
            )
        if not runtime_people:
            runtime_people.append((-1, None, None, None))
        visual_values = [
            visual_box or (0.0, 0.0, 0.0, 0.0)
            for _, _, visual_box, _ in runtime_people
        ]
        visual_tensor = torch.tensor(
            [visual_values], device=self.device, dtype=torch.float32
        )
        actor_mask = torch.tensor(
            [
                [
                    float(visual_box is not None and geometry_box is not None)
                    for _, _, visual_box, geometry_box in runtime_people
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            features = self.adapter.features_from_captured(
                visual_tensor, actor_mask
            )
        self._sync()
        feature_done = time.perf_counter()
        person_features: dict[int, _OnlinePersonFeature] = {}
        for person_index, (key, _, _, geometry_box) in enumerate(
            runtime_people
        ):
            person_features[key] = _OnlinePersonFeature(
                roi_feature=features.roi_features[0, person_index],
                geometry_box=torch.tensor(
                    geometry_box or (0.0, 0.0, 0.0, 0.0),
                    device=self.device,
                    dtype=torch.float32,
                ),
                actor_valid=actor_mask[0, person_index],
            )
        self._history.append(
            _OnlineFrameFeature(
                global_feature=features.global_features[0],
                people=person_features,
            )
        )
        person_keys = [value[0] for value in runtime_people]
        inputs = self._history_inputs(person_keys)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            outputs = self.model(*inputs)
        self._sync()
        confirmer_done = time.perf_counter()
        attention_weights = outputs.get("fusion_attention_weights")
        global_attention: float | None = None
        roi_attention: float | None = None
        if attention_weights is not None:
            global_attention, roi_attention = _attention_masses(
                attention_weights,
                use_global=self.model.use_global,
                use_roi=self.model.use_roi,
                visual_slots=len(self.model.visual_offsets),
            )
        selected_person_index = int(
            outputs["selected_person_index"][0].item()
        )
        actor = runtime_people[selected_person_index][1]
        person_scores = outputs["person_fall_logits"][0].float().sigmoid()
        falling_scores = outputs["person_falling_logits"][0].float().sigmoid()
        fallen_scores = outputs["person_fallen_logits"][0].float().sigmoid()
        geometry_scores = (
            outputs["person_geometry_fall_logits"][0].float().sigmoid()
        )
        person_predictions = tuple(
            OnlinePersonPrediction(
                track_id=person.track_id,
                bbox=person.bbox_xyxy,
                score=float(person_scores[person_index].item()),
                falling_score=float(falling_scores[person_index].item()),
                fallen_score=float(fallen_scores[person_index].item()),
                geometry_score=float(geometry_scores[person_index].item()),
            )
            for person_index, (_, person, _, _) in enumerate(runtime_people)
            if person is not None
        )
        return OnlinePrediction(
            sample_index=int(sample_index),
            timestamp_s=float(sample_index / self.effective_fps),
            score=float(outputs["fall_logits"].float().sigmoid().item()),
            geometry_score=float(
                outputs["geometry_fall_logits"].float().sigmoid().item()
            ),
            global_attention=global_attention,
            roi_attention=roi_attention,
            actor_track_id=None if actor is None else actor.track_id,
            actor_box=None if actor is None else actor.bbox_xyxy,
            people=person_predictions,
            detector_ms=(detector_done - algorithm_started) * 1000.0,
            tracking_ms=(tracking_done - detector_done) * 1000.0,
            feature_pool_ms=(feature_done - tracking_done) * 1000.0,
            confirmer_ms=(confirmer_done - feature_done) * 1000.0,
            algorithm_ms=(confirmer_done - algorithm_started) * 1000.0,
        )

    def close(self) -> None:
        self.adapter.close()


def _sampled_source_indices(
    frame_count: int,
    source_fps: float,
    target_fps: float,
) -> tuple[int, ...]:
    if min(frame_count, source_fps, target_fps) <= 0:
        raise ValueError("Frame count and frame rates must be positive")
    duration_s = (frame_count - 1) / source_fps
    count = int(math.floor(duration_s * target_fps + 1e-9)) + 1
    return tuple(
        min(frame_count - 1, int(round(index * source_fps / target_fps)))
        for index in range(count)
    )


def run_video(
    runtime: OnlineFygrRuntime,
    video_path: Path,
    *,
    path: str,
    label: int | None,
    alarm_threshold: float,
    warmup_samples: int,
) -> tuple[VideoOnlineResult, list[OnlinePrediction]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Invalid video properties: {video_path}")
    source_indices = _sampled_source_indices(
        frame_count, source_fps, runtime.effective_fps
    )
    runtime.reset_stream()
    predictions: list[OnlinePrediction] = []
    decode_ms = 0.0
    next_sample = 0
    wall_started = time.perf_counter()
    try:
        for frame_index in range(frame_count):
            decode_started = time.perf_counter()
            success, frame = capture.read()
            decode_ms += (time.perf_counter() - decode_started) * 1000.0
            if not success:
                raise ValueError(
                    f"Video ended at frame {frame_index}/{frame_count}: {video_path}"
                )
            while (
                next_sample < len(source_indices)
                and source_indices[next_sample] == frame_index
            ):
                predictions.append(
                    runtime.process_frame(frame, sample_index=next_sample)
                )
                next_sample += 1
    finally:
        capture.release()
    wall_s = time.perf_counter() - wall_started
    if next_sample != len(source_indices):
        raise RuntimeError(f"Did not process every sampled frame: {video_path}")
    scores = [prediction.score for prediction in predictions]
    first_alarm = next(
        (
            prediction.timestamp_s
            for prediction in predictions
            if prediction.score >= alarm_threshold
        ),
        None,
    )
    # Warm-up is excluded only from aggregate latency, never from alarm scores.
    if warmup_samples > 0:
        predictions[:warmup_samples] = [
            replace(prediction, timed=False)
            for prediction in predictions[:warmup_samples]
        ]
    return (
        VideoOnlineResult(
            path=path,
            label=label,
            source_fps=source_fps,
            source_frames=frame_count,
            sampled_frames=len(predictions),
            video_score=max(scores, default=0.0),
            first_alarm_s=first_alarm,
            decode_ms=decode_ms,
            wall_s=wall_s,
        ),
        predictions,
    )


def _binary_metrics(
    results: Sequence[VideoOnlineResult], threshold: float
) -> dict[str, float | int]:
    labeled = [result for result in results if result.label is not None]
    tp = sum(
        result.label == 1 and result.video_score >= threshold
        for result in labeled
    )
    fp = sum(
        result.label == 0 and result.video_score >= threshold
        for result in labeled
    )
    tn = sum(result.label == 0 and result.video_score < threshold for result in labeled)
    fn = sum(result.label == 1 and result.video_score < threshold for result in labeled)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "accuracy": (tp + tn) / len(labeled) if labeled else 0.0,
    }


def _ranking_metrics(results: Sequence[VideoOnlineResult]) -> dict[str, float]:
    labeled = [result for result in results if result.label is not None]
    empty = {
        "average_precision": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "mean_precision": 0.0,
    }
    if not labeled:
        return empty
    records = sorted(
        ((result.video_score, bool(result.label)) for result in labeled),
        key=lambda item: item[0],
        reverse=True,
    )
    positive_count = sum(int(label) for _, label in records)
    if positive_count == 0:
        return empty

    curve: list[tuple[float, float]] = []
    true_positives = false_positives = 0
    average_precision = previous_recall = 0.0
    index = 0
    while index < len(records):
        score = records[index][0]
        group_true_positives = group_size = 0
        while index < len(records) and records[index][0] == score:
            group_true_positives += int(records[index][1])
            group_size += 1
            index += 1
        true_positives += group_true_positives
        false_positives += group_size - group_true_positives
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        curve.append((recall, precision))

    def precision_at_recall(target: float) -> float:
        return max(
            (precision for recall, precision in curve if recall >= target),
            default=0.0,
        )

    p90 = precision_at_recall(0.90)
    p95 = precision_at_recall(0.95)
    return {
        "average_precision": average_precision,
        "p90": p90,
        "p95": p95,
        "mean_precision": (p90 + p95) * 0.5,
    }


def _manifest_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_video is not None:
        return [
            {
                "path": args.input_video.stem,
                "label": args.label,
                "video_path": str(args.input_video),
            }
        ]
    if args.manifest is None:
        raise ValueError("Either --input-video or --manifest is required")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("domain") == args.domain and row.get("split") == args.split
        ]
    return _stratified_sample(rows, args.max_videos, seed=args.seed)


def _stratified_sample(
    rows: Sequence[dict[str, Any]],
    limit: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    values = list(rows)
    if not limit or limit >= len(values):
        return values
    groups: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in values:
        groups[int(row["label"])].append(row)
    positive_count = round(limit * len(groups[1]) / len(values))
    if groups[0] and groups[1] and limit >= 2:
        positive_count = min(max(positive_count, 1), limit - 1)
    targets = {1: positive_count, 0: limit - positive_count}
    generator = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        count = min(targets[label], len(groups[label]))
        selected.extend(generator.sample(groups[label], count))
    if len(selected) < limit:
        selected_ids = {id(row) for row in selected}
        remaining = [row for row in values if id(row) not in selected_ids]
        selected.extend(generator.sample(remaining, limit - len(selected)))
    return sorted(selected, key=lambda row: str(row["path"]).casefold())


def _video_path(row: dict[str, Any], args: argparse.Namespace) -> Path:
    explicit = str(row.get("video_path", "")).strip()
    if explicit:
        return Path(explicit)
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for relative manifest videos")
    relative = Path(*str(row["path"]).split("/"))
    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        candidate = args.dataset_root / relative.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve video for {row['path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-video", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--domain", default="omnifall")
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", type=int, choices=(0, 1))
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alarm-threshold", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--quantize", type=int, default=16)
    parser.add_argument("--max-actor-missed", type=int, default=2)
    parser.add_argument("--max-people", type=int, default=2)
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.alarm_threshold <= 1.0:
        raise ValueError("alarm-threshold must be in [0, 1]")
    if (
        min(args.log_interval, args.warmup_samples + 1, args.max_people) <= 0
        or args.max_videos < 0
    ):
        raise ValueError("Invalid logging, warmup, or video limit")
    device = _device(args.device)
    output_dir = args.output_dir or (
        Path("artifacts/evaluations/fygr_online")
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records = _manifest_records(args)
    if not records:
        raise ValueError("No videos matched the requested source/domain/split")
    runtime = OnlineFygrRuntime(
        checkpoint_path=args.checkpoint,
        weights_path=args.weights,
        device=device,
        amp=args.amp,
        confidence=args.confidence,
        iou=args.iou,
        tracker=args.tracker,
        quantize=args.quantize,
        max_actor_missed=args.max_actor_missed,
        max_people=args.max_people,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    videos: list[VideoOnlineResult] = []
    predictions: list[tuple[str, OnlinePrediction]] = []
    try:
        for index, row in enumerate(records, start=1):
            result, frame_predictions = run_video(
                runtime,
                _video_path(row, args),
                path=str(row["path"]),
                label=None if row.get("label") in (None, "") else int(row["label"]),
                alarm_threshold=args.alarm_threshold,
                warmup_samples=args.warmup_samples if index == 1 else 0,
            )
            videos.append(result)
            predictions.extend((result.path, item) for item in frame_predictions)
            if index % args.log_interval == 0 or index == len(records):
                print(
                    f"videos={index}/{len(records)} score={result.video_score:.4f} "
                    f"sampled={result.sampled_frames} wall_s={result.wall_s:.2f}",
                    flush=True,
                )
    finally:
        runtime.close()

    timed = [item for _, item in predictions if item.timed]
    latency: dict[str, Any] = {"timed_frames": len(timed)}
    for field in (
        "detector_ms",
        "tracking_ms",
        "feature_pool_ms",
        "confirmer_ms",
        "algorithm_ms",
    ):
        values = [float(getattr(item, field)) for item in timed]
        latency[field] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
        }
    algorithm_mean = latency["algorithm_ms"]["mean"]
    latency["algorithm_fps"] = 1000.0 / algorithm_mean if algorithm_mean else 0.0
    latency["decode_ms_total"] = sum(result.decode_ms for result in videos)
    latency["wall_s_total"] = sum(result.wall_s for result in videos)
    latency["peak_cuda_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "weights": str(args.weights),
        "domain": args.domain,
        "split": args.split,
        "videos": len(videos),
        "sampled_frames": sum(result.sampled_frames for result in videos),
        "alarm_threshold": args.alarm_threshold,
        "ranking": _ranking_metrics(videos),
        "fixed_threshold": _binary_metrics(videos, args.alarm_threshold),
        "latency": latency,
        "parameters": {
            "detector": runtime.detector_parameter_count,
            "detector_runtime_fused": runtime.fused_detector_parameter_count,
            "confirmer_including_frozen_geometry": runtime.confirmer_parameter_count,
            "total": (
                runtime.detector_parameter_count
                + runtime.confirmer_parameter_count
            ),
            "total_runtime_fused": (
                runtime.fused_detector_parameter_count
                + runtime.confirmer_parameter_count
            ),
        },
        "sample_seed": args.seed,
        "causal_actor_selection": True,
        "max_people": args.max_people,
        "person_level_outputs": True,
        "disk_feature_cache": False,
    }
    with (output_dir / "videos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(videos[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in videos)
    with (output_dir / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["path", *asdict(predictions[0][1]).keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {"path": path, **asdict(prediction)}
            for path, prediction in predictions
        )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"output={output_dir}", flush=True)
    return report


def main() -> None:
    run_benchmark(build_parser().parse_args())


if __name__ == "__main__":
    main()
