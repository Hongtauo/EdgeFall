"""FastAPI application for the EdgeFall-ST real-time demo."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ALARM_THRESHOLD, CHECKPOINT, DEMO_CLIPS, FRONTEND_DIST, YOLO_WEIGHTS
from .schemas import SettingsUpdate, StreamControl
from .stream_service import StreamService


service = StreamService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    yield
    await service.close()


app = FastAPI(
    title="EdgeFall-ST 实时跌倒检测演示",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "status": service.status, "engine": service.snapshot()["system"]}


@app.get("/api/config")
async def config() -> dict:
    return {
        "model": "EdgeFall-ST HN3-B-R Seed-1 Epoch 8",
        "version": "v3.6-md-gcva-hn3br-final",
        "default_threshold": ALARM_THRESHOLD,
        "parameters": 12_212_714,
        "runtime": {
            "sample_fps": 10,
            "detector_confidence": 0.10,
            "nms_iou": 0.30,
            "max_people": 2,
            "tracker": "ByteTrack",
        },
        "checkpoint_available": CHECKPOINT.is_file(),
        "weights_available": YOLO_WEIGHTS.is_file(),
        "clips": [
            {
                "index": index,
                "title": clip.title,
                "scene": clip.scene,
                "dataset": clip.dataset,
                "expected": clip.expected,
            }
            for index, clip in enumerate(DEMO_CLIPS)
        ],
    }


@app.get("/api/state")
async def state() -> dict:
    return service.snapshot()


@app.post("/api/stream/control")
async def stream_control(command: StreamControl) -> dict:
    if command.action == "start":
        return await service.start()
    if command.action == "pause":
        return await service.pause()
    if command.action == "resume":
        return await service.start()
    return await service.restart()


@app.patch("/api/settings")
async def settings(update: SettingsUpdate) -> dict:
    return await service.update_settings(
        threshold=update.threshold,
        playback_speed=update.playback_speed,
    )


@app.get("/api/events")
async def events() -> dict:
    return {"items": list(service.events)}


@app.post("/api/events/{event_id}/acknowledge")
async def acknowledge_event(event_id: str) -> dict:
    event = service.acknowledge(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    await service._publish_snapshot()
    return event


@app.get("/api/stream/mjpeg")
async def mjpeg_stream() -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        sequence = -1
        while True:
            sequence, jpeg = await service.wait_for_frame(sequence)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = service.subscribe()
    try:
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                payload = {"type": "heartbeat", "state": service.snapshot()}
            await websocket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        service.unsubscribe(queue)


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
