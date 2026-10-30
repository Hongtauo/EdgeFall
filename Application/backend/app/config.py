"""Application configuration and demo clip metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = APPLICATION_ROOT.parent
DEMO_ROOT = FINAL_ROOT / "demo"
SOURCE_ROOT = DEMO_ROOT / "source"
INFERENCE_ROOT = DEMO_ROOT / "inference"
MODEL_ROOT = FINAL_ROOT / "models"
FRONTEND_DIST = APPLICATION_ROOT / "frontend" / "dist"

# Keep third-party runtime state inside the application. This avoids relying on
# a writable roaming-profile directory on locked-down demo machines.
RUNTIME_ROOT = APPLICATION_ROOT / ".runtime"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))


@dataclass(frozen=True)
class DemoClip:
    filename: str
    title: str
    scene: str
    expected: str
    dataset: str

    @property
    def path(self) -> Path:
        return SOURCE_ROOT / self.filename


DEMO_CLIPS = (
    # Lead with real captured datasets. The two synthetic OmniFall examples are
    # intentionally spaced late in the loop so they remain edge-case coverage,
    # rather than dominating the first impression of the demo.
    DemoClip("15_urfd_side_fall.mp4", "侧向跌倒", "办公室 · URFD", "fall", "UR Fall Detection"),
    DemoClip("06_fvd_no_fall.mp4", "正常活动", "公共区域 · 短片段", "normal", "Fall Video Dataset"),
    DemoClip("12_caucafall_forward.avi", "向前跌倒", "居家 · 广角监控", "fall", "CAUCAFall"),
    DemoClip("17_urfd_bending_walk.mp4", "弯腰行走", "卧室 · 易混淆", "normal", "UR Fall Detection"),
    DemoClip("05_fvd_fall.mp4", "走廊跌倒", "公共区域 · 侧面视角", "fall", "Fall Video Dataset"),
    DemoClip("13_caucafall_hop.avi", "跳跃动作", "居家 · 快速运动", "normal", "CAUCAFall"),
    DemoClip("16_urfd_chair_fall.mp4", "座椅旁跌倒", "办公室 · URFD", "fall", "UR Fall Detection"),
    DemoClip("10_fvd_bed_activity.mp4", "床上活动", "卧室 · 俯视监控", "normal", "Fall Video Dataset"),
    DemoClip("01_omnifall_fall.mp4", "客厅跌倒", "合成场景 · 正面视角", "fall", "OmniFall"),
    DemoClip("18_urfd_tie_shoe.mp4", "蹲下系鞋", "办公室 · 易混淆", "normal", "UR Fall Detection"),
    DemoClip("11_fvd_gym_fall.mp4", "训练场跌倒", "体育馆 · 远距离", "fall", "Fall Video Dataset"),
    DemoClip("14_caucafall_pickup.avi", "弯腰拾物", "居家 · 易混淆", "normal", "CAUCAFall"),
    DemoClip("21_caucafall_backward.avi", "向后跌倒", "居家 · 广角监控", "fall", "CAUCAFall"),
    DemoClip("19_fvd_no_fall_2.mp4", "室内日常活动", "真实监控 · 远距离", "normal", "Fall Video Dataset"),
    DemoClip("20_fvd_gym_fall_2.mp4", "训练场跌倒二", "体育馆 · 远距离", "fall", "Fall Video Dataset"),
    DemoClip("22_caucafall_sit_down.avi", "正常坐下二", "居家 · 易混淆", "normal", "CAUCAFall"),
    DemoClip("23_urfd_daily_walk.mp4", "室内行走", "卧室 · URFD", "normal", "UR Fall Detection"),
    DemoClip("04_omnifall_sit_down_adl.mp4", "正常坐下", "合成场景 · 易混淆", "normal", "OmniFall"),
)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


ENGINE_MODE = os.getenv("EDGEFALL_ENGINE", "auto").strip().lower()
DEVICE = os.getenv("EDGEFALL_DEVICE", "auto").strip().lower()
ALARM_THRESHOLD = min(1.0, max(0.0, env_float("EDGEFALL_THRESHOLD", 0.7094)))
# The checkpoint feature contract is trained and evaluated at 10 sampled FPS.
STREAM_FPS = 10.0
CHECKPOINT = MODEL_ROOT / "hn3br_epoch008_inference.pt"
YOLO_WEIGHTS = MODEL_ROOT / "yolo26s.pt"
