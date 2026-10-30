"""Build a review package for multi-person OmniFall training videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import cv2
import numpy as np

from edgefall_st.data.dataset import _additional_actor_paths


@dataclass(frozen=True)
class AuditRecord:
    path: str
    action: str
    frame_count: int
    primary_frames: int
    secondary_frames: int
    coexist_frames: int
    distinct_coexist_frames: int
    max_people_detected: int
    median_primary_secondary_iou: float
    mean_secondary_confidence: float
    primary_track_ids: str
    secondary_track_ids: str
    auto_group: str
    source_video: str
    review_video: str
    thumbnail: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _normalized_box(row: dict[str, Any], key: str) -> list[float] | None:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = np.asarray(value, dtype=np.float64)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box.tolist()


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _track_ids(rows: Sequence[dict[str, Any]]) -> str:
    values = {
        int(value)
        for row in rows
        if (value := row.get("actor_source_track_id")) is not None
        and int(value) >= 0
    }
    return "|".join(str(value) for value in sorted(values))


def analyze_video(
    *,
    path: str,
    track_frames: Sequence[dict[str, Any]],
    primary_targets: Sequence[dict[str, Any]],
    dataset_root: Path,
    min_coexist_frames: int,
    distinct_iou_threshold: float,
) -> tuple[AuditRecord | None, tuple[dict[str, Any], ...]]:
    if len(track_frames) != len(primary_targets):
        raise ValueError(f"Track/actor length mismatch for {path}")
    secondary_targets = _additional_actor_paths(
        track_frames, primary_targets, 1
    )[0]

    primary_frames = 0
    secondary_frames = 0
    coexist_frames = 0
    distinct_frames = 0
    max_people = 0
    pair_ious: list[float] = []
    secondary_confidences: list[float] = []
    for frame, primary, secondary in zip(
        track_frames, primary_targets, secondary_targets, strict=True
    ):
        detections = frame.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"Invalid detections for {path}")
        max_people = max(max_people, len(detections))
        primary_box = _normalized_box(primary, "actor_bbox_xyxy_norm")
        secondary_box = _normalized_box(
            secondary, "actor_bbox_xyxy_norm"
        )
        primary_frames += primary_box is not None
        secondary_frames += secondary_box is not None
        if secondary_box is not None:
            confidence = secondary.get("actor_confidence")
            if confidence is not None:
                secondary_confidences.append(float(confidence))
        if primary_box is None or secondary_box is None:
            continue
        coexist_frames += 1
        iou = _box_iou(primary_box, secondary_box)
        pair_ious.append(iou)
        distinct_frames += iou <= distinct_iou_threshold

    if secondary_frames < min_coexist_frames or coexist_frames < min_coexist_frames:
        return None, secondary_targets

    pair_median = median(pair_ious) if pair_ious else 0.0
    auto_group = (
        "spatially_distinct"
        if distinct_frames >= min_coexist_frames
        else "overlap_review"
    )
    flattened = path.replace("/", "__")
    action = path.split("/", maxsplit=1)[0]
    record = AuditRecord(
        path=path,
        action=action,
        frame_count=len(track_frames),
        primary_frames=primary_frames,
        secondary_frames=secondary_frames,
        coexist_frames=coexist_frames,
        distinct_coexist_frames=distinct_frames,
        max_people_detected=max_people,
        median_primary_secondary_iou=pair_median,
        mean_secondary_confidence=(
            float(np.mean(secondary_confidences))
            if secondary_confidences
            else 0.0
        ),
        primary_track_ids=_track_ids(primary_targets),
        secondary_track_ids=_track_ids(secondary_targets),
        auto_group=auto_group,
        source_video=str(dataset_root / f"{path}.mp4"),
        review_video=f"videos/{flattened}.mp4",
        thumbnail=f"thumbnails/{flattened}.jpg",
    )
    return record, secondary_targets


def _draw_box(
    frame: np.ndarray,
    box: Sequence[float],
    *,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    left = int(round(float(box[0]) * width))
    top = int(round(float(box[1]) * height))
    right = int(round(float(box[2]) * width))
    bottom = int(round(float(box[3]) * height))
    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    right = min(max(right, left + 1), width - 1)
    bottom = min(max(bottom, top + 1), height - 1)
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    font_scale = max(0.42, min(width, height) / 1100.0)
    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    label_top = max(0, top - text_size[1] - baseline - 4)
    cv2.rectangle(
        frame,
        (left, label_top),
        (min(width - 1, left + text_size[0] + 6), top),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (left + 3, max(text_size[1] + 1, top - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_header(frame: np.ndarray, path: str, frame_index: int) -> None:
    height, width = frame.shape[:2]
    header_height = max(42, int(round(height * 0.075)))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, header_height), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
    font_scale = max(0.42, min(width, height) / 1150.0)
    cv2.putText(
        frame,
        f"{path}  frame={frame_index}",
        (10, int(header_height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "RED=TRAIN PRIMARY   YELLOW=SECONDARY   GREEN=OTHER DETECTIONS",
        (10, int(header_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (220, 225, 230),
        1,
        cv2.LINE_AA,
    )


def render_review_video(
    *,
    record: AuditRecord,
    track_frames: Sequence[dict[str, Any]],
    primary_targets: Sequence[dict[str, Any]],
    secondary_targets: Sequence[dict[str, Any]],
    output_root: Path,
    frame_step: int,
    max_width: int,
    overwrite: bool,
) -> None:
    output_path = output_root / record.review_video
    thumbnail_path = output_root / record.thumbnail
    if output_path.exists() and thumbnail_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(record.source_video)
    if not capture.isOpened():
        raise OSError(f"Could not open {record.source_video}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise ValueError(f"Invalid video size for {record.path}")
    scale = min(1.0, max_width / source_width)
    output_width = max(2, int(round(source_width * scale)) // 2 * 2)
    output_height = max(2, int(round(source_height * scale)) // 2 * 2)
    output_fps = (
        source_fps / frame_step
        if math.isfinite(source_fps) and source_fps > 0.0
        else 8.0
    )
    output_fps = max(1.0, output_fps)
    temporary = output_path.with_name(f"{output_path.stem}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create {temporary}")

    thumbnail_frame: np.ndarray | None = None
    thumbnail_iou: float | None = None
    fallback_thumbnail: np.ndarray | None = None
    middle = len(track_frames) // 2
    try:
        for frame_index, (track, primary, secondary) in enumerate(
            zip(
                track_frames,
                primary_targets,
                secondary_targets,
                strict=True,
            )
        ):
            ok, frame = capture.read()
            if not ok:
                raise OSError(
                    f"Video ended at frame {frame_index}: {record.path}"
                )
            if frame_index % frame_step != 0:
                continue
            if (output_width, output_height) != (source_width, source_height):
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            detections = track.get("detections")
            if not isinstance(detections, list):
                raise ValueError(f"Invalid detections for {record.path}")
            for detection in detections:
                box = _normalized_box(detection, "bbox_xyxy_norm")
                if box is None:
                    continue
                track_id = detection.get("track_id", "?")
                confidence = float(detection.get("confidence", 0.0))
                _draw_box(
                    frame,
                    box,
                    color=(62, 170, 78),
                    label=f"OTHER id={track_id} conf={confidence:.2f}",
                    thickness=2,
                )
            secondary_box = _normalized_box(
                secondary, "actor_bbox_xyxy_norm"
            )
            if secondary_box is not None:
                secondary_id = secondary.get("actor_source_track_id", "?")
                _draw_box(
                    frame,
                    secondary_box,
                    color=(0, 205, 235),
                    label=f"SECONDARY id={secondary_id}",
                    thickness=3,
                )
            primary_box = _normalized_box(primary, "actor_bbox_xyxy_norm")
            if primary_box is not None:
                primary_id = primary.get("actor_source_track_id", "?")
                _draw_box(
                    frame,
                    primary_box,
                    color=(38, 38, 225),
                    label=f"TRAIN PRIMARY id={primary_id}",
                    thickness=4,
                )
            _draw_header(frame, record.path, frame_index)
            writer.write(frame)
            if primary_box is not None and secondary_box is not None:
                pair_iou = _box_iou(primary_box, secondary_box)
                if thumbnail_iou is None or pair_iou < thumbnail_iou:
                    thumbnail_frame = frame.copy()
                    thumbnail_iou = pair_iou
            elif fallback_thumbnail is None and frame_index >= middle:
                fallback_thumbnail = frame.copy()
    finally:
        capture.release()
        writer.release()

    if not temporary.exists() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise OSError(f"Empty review video for {record.path}")
    temporary.replace(output_path)
    if thumbnail_frame is None:
        thumbnail_frame = fallback_thumbnail
    if thumbnail_frame is None:
        raise OSError(f"Could not choose thumbnail for {record.path}")
    if not cv2.imwrite(str(thumbnail_path), thumbnail_frame):
        raise OSError(f"Could not write {thumbnail_path}")


def _write_csv(path: Path, records: Sequence[AuditRecord]) -> None:
    fields = [
        *AuditRecord.__dataclass_fields__,
        "manual_is_multi",
        "manual_primary_correct",
        "manual_correct_primary",
        "manual_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **asdict(record),
                    "manual_is_multi": "",
                    "manual_primary_correct": "",
                    "manual_correct_primary": "",
                    "manual_notes": "",
                }
            )


def _write_html(
    path: Path,
    records: Sequence[AuditRecord],
    *,
    video_directory: str,
) -> None:
    data = json.dumps(
        [asdict(record) for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    document = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniFall 训练集多人审核</title>
<style>
:root{color-scheme:light;font-family:Arial,"Microsoft YaHei",sans-serif;background:#f4f6f8;color:#18202a}
*{box-sizing:border-box}html,body{height:100%;overflow:hidden}body{margin:0}button,input,textarea,select{font:inherit}
header{height:58px;background:#fff;border-bottom:1px solid #d9dee5;display:flex;align-items:center;gap:18px;padding:0 18px}
header h1{font-size:18px;margin:0}#progress{color:#52606d;font-size:13px}
.layout{display:grid;grid-template-columns:340px minmax(0,1fr);height:calc(100dvh - 58px);min-height:0;overflow:hidden}
aside{background:#fff;border-right:1px solid #d9dee5;display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}
.filters{padding:12px;border-bottom:1px solid #e4e8ed;display:grid;grid-template-columns:1fr 120px;gap:8px}
.filters input,.filters select{height:36px;border:1px solid #bcc5cf;border-radius:4px;padding:0 9px;min-width:0}
#list{overflow:auto;overscroll-behavior:contain;min-height:0;flex:1}.item{width:100%;border:0;border-bottom:1px solid #edf0f3;background:#fff;text-align:left;padding:10px 12px;cursor:pointer}
.item:hover{background:#f2f6f8}.item.active{background:#dfeef2;border-left:4px solid #16758a;padding-left:8px}
.item .name{font-weight:700;font-size:13px;overflow-wrap:anywhere}.item .meta{font-size:12px;color:#66717d;margin-top:4px}
main{padding:16px 20px;overflow:auto;overscroll-behavior:contain;min-height:0}.titlebar{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:10px}
#title{font-size:18px;margin:0;overflow-wrap:anywhere}.stats{font-size:12px;color:#596673;margin-top:5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px}.dot{display:inline-block;width:10px;height:10px;margin-right:5px}.red{background:#e12626}.yellow{background:#ebcd00}.green{background:#3eaa4e}
video{display:block;width:100%;max-height:68vh;background:#111;aspect-ratio:16/9}.review{margin-top:14px;border-top:1px solid #d9dee5;padding-top:14px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.row strong{width:118px;font-size:13px}
.choice{height:34px;border:1px solid #aeb8c3;border-radius:4px;background:#fff;padding:0 13px;cursor:pointer}.choice.selected{background:#176f82;color:#fff;border-color:#176f82}
textarea{width:100%;min-height:72px;border:1px solid #aeb8c3;border-radius:4px;padding:8px;resize:vertical}
.actions{display:flex;gap:8px;justify-content:flex-end;margin-top:10px}.command{height:36px;border:1px solid #176f82;background:#176f82;color:#fff;border-radius:4px;padding:0 14px;cursor:pointer}.command.secondary{background:#fff;color:#176f82}
@media(max-width:820px){.layout{grid-template-columns:1fr;grid-template-rows:300px minmax(0,1fr);height:calc(100dvh - 58px)}aside{height:auto;border-right:0;border-bottom:1px solid #d9dee5}main{padding:12px}.titlebar{display:block}.legend{margin-top:8px}}
</style>
</head>
<body>
<header><h1>OmniFall 训练集多人审核</h1><span id="progress"></span></header>
<div class="layout">
<aside>
  <div class="filters"><input id="search" placeholder="搜索视频"><select id="group"><option value="">全部候选</option><option value="spatially_distinct">空间分离候选</option><option value="overlap_review">重叠框候选</option></select></div>
  <div id="list"></div>
</aside>
<main>
  <div class="titlebar"><div><h2 id="title"></h2><div id="stats" class="stats"></div></div><div class="legend"><span><i class="dot red"></i>训练主人物</span><span><i class="dot yellow"></i>第二人物路径</span><span><i class="dot green"></i>其他检测</span></div></div>
  <video id="video" controls preload="metadata"></video>
  <section class="review">
    <div class="row"><strong>是否真实多人</strong><button class="choice" data-field="is_multi" data-value="yes">多人</button><button class="choice" data-field="is_multi" data-value="no">单人/重复框</button><button class="choice" data-field="is_multi" data-value="uncertain">不确定</button></div>
    <div class="row"><strong>主人物是否正确</strong><button class="choice" data-field="primary_correct" data-value="yes">正确</button><button class="choice" data-field="primary_correct" data-value="no">错误</button><button class="choice" data-field="primary_correct" data-value="uncertain">不确定</button></div>
    <textarea id="notes" placeholder="正确主人物、错误原因或其他备注"></textarea>
    <div class="actions"><button id="previous" class="command secondary">上一个</button><button id="next" class="command secondary">下一个</button><button id="export" class="command">导出审核 CSV</button></div>
  </section>
</main>
</div>
<script>
const records=__DATA__;
const videoDirectory=__VIDEO_DIRECTORY__;
const key="omnifall_train_multi_person_review_v1";
const decisions=JSON.parse(localStorage.getItem(key)||"{}");
let filtered=[...records],current=0;
const list=document.querySelector("#list"),video=document.querySelector("#video"),title=document.querySelector("#title"),stats=document.querySelector("#stats"),notes=document.querySelector("#notes");
function save(){localStorage.setItem(key,JSON.stringify(decisions));updateProgress()}
function decision(){const r=filtered[current];if(!r)return null;return decisions[r.path]||(decisions[r.path]={})}
function updateProgress(){const done=records.filter(r=>decisions[r.path]?.is_multi).length;document.querySelector("#progress").textContent=`已审核 ${done} / ${records.length}`}
function renderList(){list.innerHTML="";filtered.forEach((r,i)=>{const b=document.createElement("button");b.className="item"+(i===current?" active":"");b.innerHTML=`<div class="name">${r.path}</div><div class="meta">${r.auto_group} · 共现 ${r.coexist_frames} 帧 · ${decisions[r.path]?.is_multi||"未审核"}</div>`;b.onclick=()=>select(i);list.appendChild(b)})}
function select(i){if(!filtered.length)return;current=Math.max(0,Math.min(i,filtered.length-1));const r=filtered[current],d=decision();title.textContent=r.path;stats.textContent=`主轨迹 ID ${r.primary_track_ids||"?"} · 第二轨迹 ID ${r.secondary_track_ids||"?"} · 最大检测人数 ${r.max_people_detected} · 中位 IoU ${r.median_primary_secondary_iou.toFixed(3)}`;video.src=`${videoDirectory}/${r.review_video.split("/").pop()}`;notes.value=d.notes||"";document.querySelectorAll(".choice").forEach(b=>b.classList.toggle("selected",d[b.dataset.field]===b.dataset.value));renderList();list.children[current]?.scrollIntoView({block:"nearest"})}
function applyFilter(){const q=document.querySelector("#search").value.trim().toLowerCase(),g=document.querySelector("#group").value;filtered=records.filter(r=>(!q||r.path.toLowerCase().includes(q))&&(!g||r.auto_group===g));current=0;renderList();select(0)}
document.querySelectorAll(".choice").forEach(b=>b.onclick=()=>{const d=decision();if(!d)return;d[b.dataset.field]=b.dataset.value;save();select(current)});
notes.oninput=()=>{const d=decision();if(!d)return;d.notes=notes.value;save()};
document.querySelector("#search").oninput=applyFilter;document.querySelector("#group").onchange=applyFilter;
document.querySelector("#previous").onclick=()=>select(current-1);document.querySelector("#next").onclick=()=>select(current+1);
document.querySelector("#export").onclick=()=>{const cols=["path","auto_group","primary_track_ids","secondary_track_ids","manual_is_multi","manual_primary_correct","manual_notes"],esc=v=>`"${String(v??"").replaceAll('"','""')}"`,rows=[cols.join(",")];records.forEach(r=>{const d=decisions[r.path]||{};rows.push([r.path,r.auto_group,r.primary_track_ids,r.secondary_track_ids,d.is_multi||"",d.primary_correct||"",d.notes||""].map(esc).join(","))});const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff"+rows.join("\\r\\n")],{type:"text/csv"}));a.download="manual_review.csv";a.click();URL.revokeObjectURL(a.href)};
updateProgress();renderList();select(0);
</script>
</body></html>"""
    document = document.replace("__DATA__", data)
    document = document.replace(
        "__VIDEO_DIRECTORY__",
        json.dumps(video_directory, ensure_ascii=False),
    )
    path.write_text(document, encoding="utf-8")


def _write_readme(path: Path, records: Sequence[AuditRecord]) -> None:
    spatially_distinct = sum(
        record.auto_group == "spatially_distinct" for record in records
    )
    overlap = len(records) - spatially_distinct
    minimum_coexist = min(
        (record.coexist_frames for record in records),
        default=0,
    )
    content = f"""# OmniFall 训练集多人候选审核

候选总数：{len(records)}；空间分离候选：{spatially_distinct}；重叠框候选：{overlap}。

- 红框：当前训练真正读取的主人物路径。
- 黄框：移除主人物后，以不读取动作标签的时序选择器得到的第二人物路径。
- 绿框：YOLO26s 的其余人体检测。
- `review.csv`：全部自动统计及空白人工审核列。
- `index.html`：逐视频审核页面，结果保存在浏览器本地，点击“导出审核 CSV”保存。

自动候选只表示至少 {minimum_coexist} 帧同时存在主人物和第二路径；空间分离也可能来自局部人体误检，
不能替代人工确认。审核时重点判断两项：画面是否确有多人，以及红框是否始终落在动作主体上。
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset/omnifall/of-syn"),
    )
    parser.add_argument(
        "--tracks-root",
        type=Path,
        default=Path("artifacts/pseudo_labels/train_yolo26s_nms/tracks"),
    )
    parser.add_argument(
        "--actor-targets-root",
        type=Path,
        default=Path("artifacts/actor_targets_yolo26s/train/targets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/reviews/omnifall_train_multi_person_yolo26s"
        ),
    )
    parser.add_argument("--min-coexist-frames", type=int, default=8)
    parser.add_argument("--distinct-iou-threshold", type=float, default=0.65)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--web-video-directory", default="videos")
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.min_coexist_frames <= 0:
        raise ValueError("min-coexist-frames must be positive")
    if not 0.0 <= args.distinct_iou_threshold <= 1.0:
        raise ValueError("distinct-iou-threshold must be in [0, 1]")
    if args.frame_step <= 0 or args.max_width <= 0 or args.workers <= 0:
        raise ValueError("frame-step, max-width and workers must be positive")
    if args.max_videos < 0:
        raise ValueError("max-videos cannot be negative")
    if not args.web_video_directory or any(
        separator in args.web_video_directory for separator in ("/", "\\")
    ):
        raise ValueError("web-video-directory must be one directory name")


def run(args: argparse.Namespace) -> list[AuditRecord]:
    _validate_args(args)
    track_paths = sorted(args.tracks_root.rglob("*.jsonl"))
    if args.max_videos:
        track_paths = track_paths[: args.max_videos]
    candidates: list[
        tuple[
            AuditRecord,
            list[dict[str, Any]],
            list[dict[str, Any]],
            tuple[dict[str, Any], ...],
        ]
    ] = []
    for index, track_path in enumerate(track_paths, start=1):
        relative = track_path.relative_to(args.tracks_root)
        path = relative.with_suffix("").as_posix()
        actor_path = args.actor_targets_root / relative
        if not actor_path.exists():
            raise FileNotFoundError(actor_path)
        tracks = _read_jsonl(track_path)
        primary = _read_jsonl(actor_path)
        record, secondary = analyze_video(
            path=path,
            track_frames=tracks,
            primary_targets=primary,
            dataset_root=args.dataset_root,
            min_coexist_frames=args.min_coexist_frames,
            distinct_iou_threshold=args.distinct_iou_threshold,
        )
        if record is not None:
            candidates.append((record, tracks, primary, secondary))
        if index % 500 == 0 or index == len(track_paths):
            print(
                f"scanned={index}/{len(track_paths)} "
                f"candidates={len(candidates)}",
                flush=True,
            )

    candidates.sort(
        key=lambda item: (
            item[0].auto_group != "spatially_distinct",
            -item[0].coexist_frames,
            item[0].path,
        )
    )
    records = [item[0] for item in candidates]
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "review.csv", records)
    _write_html(
        args.output_root / "index.html",
        records,
        video_directory=args.web_video_directory,
    )
    _write_readme(args.output_root / "README.md", records)

    if args.render:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for record, tracks, primary, secondary in candidates:
                future = executor.submit(
                    render_review_video,
                    record=record,
                    track_frames=tracks,
                    primary_targets=primary,
                    secondary_targets=secondary,
                    output_root=args.output_root,
                    frame_step=args.frame_step,
                    max_width=args.max_width,
                    overwrite=args.overwrite,
                )
                futures[future] = record.path
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(futures):
                    print(
                        f"rendered={completed}/{len(futures)}",
                        flush=True,
                    )

    summary = {
        "videos_scanned": len(track_paths),
        "candidates": len(records),
        "spatially_distinct": sum(
            record.auto_group == "spatially_distinct" for record in records
        ),
        "overlap_review": sum(
            record.auto_group == "overlap_review" for record in records
        ),
        "criteria": {
            "min_coexist_frames": args.min_coexist_frames,
            "distinct_iou_threshold": args.distinct_iou_threshold,
        },
        "render": {
            "enabled": args.render,
            "frame_step": args.frame_step,
            "max_width": args.max_width,
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return records


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
