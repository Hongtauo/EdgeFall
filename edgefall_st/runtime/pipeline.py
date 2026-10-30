"""Streaming tensor-to-event pipeline for a single camera stream."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from edgefall_st.models import EdgeFallST, StreamState
from edgefall_st.runtime.events import EventConfig, FallEventManager
from edgefall_st.runtime.postprocess import postprocess_predictions
from edgefall_st.runtime.tracker import MultiObjectTracker, TrackerConfig
from edgefall_st.runtime.types import Detection, EventMessage, TrackSnapshot


@dataclass(frozen=True)
class PipelineOutput:
    timestamp_s: float
    detections: tuple[Detection, ...]
    tracks: tuple[TrackSnapshot, ...]
    events: tuple[EventMessage, ...]


class EdgeFallPipeline:
    """Compose the single learned model with parameter-free runtime logic."""

    def __init__(
        self,
        model: EdgeFallST,
        *,
        tracker_config: TrackerConfig = TrackerConfig(),
        event_config: EventConfig = EventConfig(),
        person_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        self.model = model.eval()
        self.tracker = MultiObjectTracker(tracker_config)
        self.event_manager = FallEventManager(event_config)
        self.stream_state: StreamState = model.create_stream_state()
        self.person_threshold = person_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    def reset(self) -> None:
        self.stream_state.reset()
        self.tracker.reset()
        self.event_manager.reset()

    @torch.inference_mode()
    def process(self, image: Tensor, timestamp_s: float) -> PipelineOutput:
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Pipeline input must have shape [1, 3, H, W]")
        outputs = self.model.forward_stream(image, self.stream_state)
        detections = postprocess_predictions(
            outputs,
            person_threshold=self.person_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            max_detections=self.max_detections,
        )[0]
        tracks, expired = self.tracker.update(detections)
        events = self.event_manager.update(timestamp_s, tracks, expired)
        return PipelineOutput(
            timestamp_s=timestamp_s,
            detections=tuple(detections),
            tracks=tuple(tracks),
            events=tuple(events),
        )
