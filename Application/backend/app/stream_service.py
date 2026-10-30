"""Shared demo stream producer, annotation, telemetry, and event handling."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from .config import ALARM_THRESHOLD, DEMO_CLIPS, ENGINE_MODE, STREAM_FPS
from .engines import InferenceEngine, build_engine


class StreamService:
    def __init__(self) -> None:
        self.threshold = ALARM_THRESHOLD
        self.playback_speed = 1.0
        self.status = "idle"
        self.status_message = "等待启动演示"
        self.engine: InferenceEngine | None = None
        self.latest_jpeg = self._placeholder("READY", "Start the demo stream")
        self.sequence = 0
        self.loop_count = 0
        self.clip_index = 0
        self.clip_elapsed = 0.0
        self.clip_duration = 0.0
        self.score_history: deque[dict[str, float]] = deque(maxlen=180)
        self.events: deque[dict[str, Any]] = deque(maxlen=50)
        self.telemetry: dict[str, Any] = self._base_telemetry()
        self._task: asyncio.Task[None] | None = None
        self._paused = asyncio.Event()
        self._paused.set()
        self._restart_requested = False
        self._closing = False
        self._frame_condition = asyncio.Condition()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._clip_durations = [self._read_duration(clip.path) for clip in DEMO_CLIPS]
        self._total_duration = sum(self._clip_durations)

    def _base_telemetry(self) -> dict[str, Any]:
        return {
            "type": "telemetry",
            "sequence": self.sequence,
            "server_time": datetime.now().astimezone().isoformat(),
            "system": {
                "status": self.status,
                "message": self.status_message,
                "engine_mode": None,
                "engine_label": "尚未加载",
                "device": "—",
                "model": "EdgeFall-ST HN3-B-R",
                "fallback_reason": None,
            },
            "stream": {
                "clip_index": self.clip_index,
                "clip_total": len(DEMO_CLIPS),
                "clip_name": DEMO_CLIPS[self.clip_index].title,
                "scene": DEMO_CLIPS[self.clip_index].scene,
                "dataset": DEMO_CLIPS[self.clip_index].dataset,
                "expected": DEMO_CLIPS[self.clip_index].expected,
                "elapsed_s": self.clip_elapsed,
                "duration_s": self.clip_duration,
                "progress": 0.0,
                "loop_count": self.loop_count,
                "playback_speed": self.playback_speed,
                "target_fps": STREAM_FPS,
            },
            "inference": {
                "score": 0.0,
                "threshold": self.threshold,
                "risk_level": "normal",
                "risk_label": "监测正常",
                "geometry_score": 0.0,
                "global_attention": None,
                "roi_attention": None,
                "actor_track_id": None,
                "people_count": 0,
                "people": [],
                "latency_ms": 0.0,
                "processing_fps": 0.0,
                "breakdown": {
                    "detector_ms": 0.0,
                    "tracking_ms": 0.0,
                    "feature_pool_ms": 0.0,
                    "confirmer_ms": 0.0,
                },
            },
            "active_event": self._active_event(),
            "score_history": list(self.score_history),
        }

    @staticmethod
    def _placeholder(title: str, subtitle: str) -> bytes:
        frame = np.full((720, 1280, 3), (19, 29, 31), dtype=np.uint8)
        cv2.circle(frame, (640, 310), 58, (88, 174, 148), 3, cv2.LINE_AA)
        cv2.putText(frame, title, (540, 410), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (230, 240, 237), 3, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (470, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (135, 158, 153), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Could not create placeholder frame")
        return encoded.tobytes()

    async def start(self, *, restart: bool = False) -> dict[str, Any]:
        if restart:
            self._restart_requested = True
        if self._task is None or self._task.done():
            self._closing = False
            self._paused.set()
            self.status = "loading"
            self.status_message = "正在加载推理引擎"
            self._task = asyncio.create_task(self._producer(), name="edgefall-demo-stream")
        else:
            self._paused.set()
            if self.status == "paused":
                self.status = "running"
                self.status_message = "正在接收模拟视频流"
                await self._publish_snapshot()
        return self.snapshot()

    async def pause(self) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            self._paused.clear()
            self.status = "paused"
            self.status_message = "视频流已暂停"
            await self._publish_snapshot()
        return self.snapshot()

    async def restart(self) -> dict[str, Any]:
        return await self.start(restart=True)

    async def update_settings(
        self, *, threshold: float | None, playback_speed: float | None
    ) -> dict[str, Any]:
        if threshold is not None:
            self.threshold = float(threshold)
        if playback_speed is not None:
            self.playback_speed = float(playback_speed)
        await self._publish_snapshot()
        return self.snapshot()

    async def close(self) -> None:
        self._closing = True
        self._paused.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.engine is not None:
            await asyncio.to_thread(self.engine.close)

    async def _producer(self) -> None:
        self.status = "loading"
        self.status_message = "正在加载推理引擎"
        await self._publish_snapshot()
        try:
            if self.engine is None:
                self.engine = await asyncio.to_thread(build_engine, ENGINE_MODE)
            self.status = "running"
            self.status_message = "正在接收模拟视频流"
            await self._publish_snapshot()
            while not self._closing:
                self._restart_requested = False
                for index, clip in enumerate(DEMO_CLIPS):
                    if self._restart_requested or self._closing:
                        break
                    await self._play_clip(index, clip)
                if not self._restart_requested:
                    self.loop_count += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.status = "error"
            self.status_message = f"推流中断：{error}"
            self.latest_jpeg = self._placeholder("STREAM ERROR", str(error)[:72])
            await self._publish_snapshot()

    async def _play_clip(self, index: int, clip: Any) -> None:
        capture = cv2.VideoCapture(str(clip.path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开演示视频 {clip.filename}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or STREAM_FPS
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame_time = max(0.0, (frame_count - 1) / source_fps)
        sample_count = int(last_frame_time * STREAM_FPS + 1e-9) + 1
        self.clip_index = index
        self.clip_duration = frame_count / source_fps
        self.clip_elapsed = 0.0
        emitted_alarm = False
        sample_index = 0
        await asyncio.to_thread(self.engine.reset, clip.filename)
        try:
            while sample_index < sample_count:
                await self._paused.wait()
                if self._restart_requested or self._closing:
                    return
                iteration_started = time.perf_counter()
                self.clip_elapsed = sample_index / STREAM_FPS
                source_index = min(
                    frame_count - 1,
                    int(round(self.clip_elapsed * source_fps)),
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = await asyncio.to_thread(
                    self.engine.predict, frame, sample_index, clip.filename
                )
                score = float(prediction["score"])
                if score >= self.threshold and not emitted_alarm:
                    self._add_event(clip, prediction)
                    emitted_alarm = True
                self.latest_jpeg = self._annotate(frame, prediction)
                self.sequence += 1
                history_time = (
                    self.loop_count * self._total_duration
                    + sum(self._clip_durations[:index])
                    + self.clip_elapsed
                )
                self.score_history.append({"time": history_time, "score": score})
                self.telemetry = self._make_telemetry(clip, prediction)
                await self._broadcast(self.telemetry)
                async with self._frame_condition:
                    self._frame_condition.notify_all()
                sample_index += 1
                interval = 1.0 / (STREAM_FPS * self.playback_speed)
                remaining = interval - (time.perf_counter() - iteration_started)
                if remaining > 0:
                    await asyncio.sleep(remaining)
        finally:
            capture.release()

    @staticmethod
    def _read_duration(path: Any) -> float:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or STREAM_FPS
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count / fps

    def _add_event(self, clip: Any, prediction: dict[str, Any]) -> None:
        event = {
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now().astimezone().isoformat(),
            "clip_name": clip.title,
            "scene": clip.scene,
            "score": float(prediction["score"]),
            "track_id": prediction.get("actor_track_id"),
            "status": "pending",
        }
        self.events.appendleft(event)

    def _active_event(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["status"] == "pending"), None)

    def acknowledge(self, event_id: str) -> dict[str, Any] | None:
        for event in self.events:
            if event["id"] == event_id:
                event["status"] = "acknowledged"
                event["acknowledged_at"] = datetime.now().astimezone().isoformat()
                return event
        return None

    def snapshot(self) -> dict[str, Any]:
        payload = dict(self.telemetry)
        payload["system"] = dict(payload["system"])
        payload["system"].update({"status": self.status, "message": self.status_message})
        payload["stream"] = dict(payload["stream"])
        payload["stream"].update({"playback_speed": self.playback_speed})
        payload["inference"] = dict(payload["inference"])
        payload["inference"].update({"threshold": self.threshold})
        payload["active_event"] = self._active_event()
        payload["score_history"] = list(self.score_history)
        return payload

    async def _publish_snapshot(self) -> None:
        self.telemetry = self.snapshot()
        await self._broadcast(self.telemetry)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        queue.put_nowait(self.snapshot())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def wait_for_frame(self, previous_sequence: int) -> tuple[int, bytes]:
        async with self._frame_condition:
            if self.sequence == previous_sequence:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._frame_condition.wait(), timeout=2.0)
        return self.sequence, self.latest_jpeg

    def _make_telemetry(self, clip: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        score = float(prediction["score"])
        if score >= self.threshold:
            risk_level, risk_label = "danger", "检测到跌倒"
        elif score >= self.threshold * 0.78:
            risk_level, risk_label = "warning", "需要关注"
        else:
            risk_level, risk_label = "normal", "监测正常"
        latency = float(prediction.get("algorithm_ms", 0.0))
        return {
            "type": "telemetry",
            "sequence": self.sequence,
            "server_time": datetime.now().astimezone().isoformat(),
            "system": {
                "status": self.status,
                "message": self.status_message,
                "engine_mode": self.engine.mode,
                "engine_label": self.engine.label,
                "device": self.engine.device,
                "model": "EdgeFall-ST HN3-B-R",
                "fallback_reason": self.engine.fallback_reason,
            },
            "stream": {
                "clip_index": self.clip_index,
                "clip_total": len(DEMO_CLIPS),
                "clip_name": clip.title,
                "scene": clip.scene,
                "dataset": clip.dataset,
                "expected": clip.expected,
                "elapsed_s": self.clip_elapsed,
                "duration_s": self.clip_duration,
                "progress": self.clip_elapsed / max(self.clip_duration, 1e-6),
                "loop_count": self.loop_count,
                "playback_speed": self.playback_speed,
                "target_fps": STREAM_FPS,
            },
            "inference": {
                "score": score,
                "threshold": self.threshold,
                "risk_level": risk_level,
                "risk_label": risk_label,
                "geometry_score": float(prediction.get("geometry_score", 0.0)),
                "global_attention": prediction.get("global_attention"),
                "roi_attention": prediction.get("roi_attention"),
                "actor_track_id": prediction.get("actor_track_id"),
                "people_count": len(prediction.get("people", [])),
                "people": prediction.get("people", []),
                "latency_ms": latency,
                "processing_fps": 1000.0 / latency if latency > 0 else 0.0,
                "breakdown": {
                    "detector_ms": float(prediction.get("detector_ms", 0.0)),
                    "tracking_ms": float(prediction.get("tracking_ms", 0.0)),
                    "feature_pool_ms": float(prediction.get("feature_pool_ms", 0.0)),
                    "confirmer_ms": float(prediction.get("confirmer_ms", 0.0)),
                },
            },
            "active_event": self._active_event(),
            "score_history": list(self.score_history),
        }

    def _annotate(self, frame: np.ndarray, prediction: dict[str, Any]) -> bytes:
        output = frame.copy()
        threshold = self.threshold
        score = float(prediction["score"])
        danger = score >= threshold
        color = (53, 70, 229) if danger else (118, 190, 80)
        for person in prediction.get("people", []):
            left, top, right, bottom = (int(round(value)) for value in person["bbox"])
            person_score = float(person.get("score", 0.0))
            person_color = (53, 70, 229) if person_score >= threshold else (118, 190, 80)
            cv2.rectangle(output, (left, top), (right, bottom), person_color, 3)
            label = f"ID {person.get('track_id', '-')}  {person_score:.2f}"
            cv2.rectangle(output, (left, max(0, top - 32)), (left + 165, top), person_color, -1)
            cv2.putText(output, label, (left + 8, max(22, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        banner = output.copy()
        cv2.rectangle(banner, (18, 18), (330, 92), (16, 24, 27), -1)
        cv2.addWeighted(banner, 0.78, output, 0.22, 0, output)
        cv2.circle(output, (43, 45), 7, color, -1, cv2.LINE_AA)
        cv2.putText(output, "FALL ALERT" if danger else "MONITORING", (60, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (244, 248, 247), 2, cv2.LINE_AA)
        cv2.putText(output, f"RISK {score:.3f}  /  {threshold:.3f}", (36, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (200, 213, 210), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise RuntimeError("Could not encode stream frame")
        return encoded.tobytes()
