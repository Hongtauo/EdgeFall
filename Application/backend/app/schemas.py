"""API request models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StreamControl(BaseModel):
    action: Literal["start", "pause", "resume", "restart"]


class SettingsUpdate(BaseModel):
    threshold: float | None = Field(default=None, ge=0.1, le=0.95)
    playback_speed: float | None = Field(default=None, ge=0.5, le=2.0)
