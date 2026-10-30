"""Extract frozen YOLO features for selected V3.5 external-domain videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from edgefall_st.features.adapter import UltralyticsYOLOFeatureAdapter
from edgefall_st.features.cache import FrozenFeatureCacheWriter
from edgefall_st.features.cache import roi_feature_shape
from external_eval.caucafall.common import CaucaVideo, discover_videos
from external_eval.fall_video_dataset.common import resampled_source_indices
from external_eval.fall_video_dataset.evaluate_v3_yolo import (
    read_actor_box_sequence,
)
from external_eval.fygr_common import (
    FrameItem,
    extract_external_sequence,
)


ACTION_IDS = {
    "Hop": 15,
    "Kneel": 11,
    "Pick up object": 9,
    "Sit down": 3,
    "Walk": 0,
}


@dataclass(frozen=True)
class CacheVideo:
    path: str
    domain: str
    split: str
    label: bool
    label_name: str
    video_path: Path


class _FeatureRuntime:
    def __init__(
        self,
        *,
        adapter: UltralyticsYOLOFeatureAdapter,
        contract: Mapping[str, object],
        device: torch.device,
        amp: bool,
    ) -> None:
        arrays = contract["arrays"]
        if not isinstance(arrays, Mapping):
            raise TypeError("Feature contract arrays must be a mapping")
        self.adapter = adapter
        self.device = device
        self.visual_input_size = tuple(
            int(value) for value in contract["visual_input_size"]
        )
        self.geometry_input_size = tuple(
            int(value) for value in contract["geometry_input_size"]
        )
        self.global_shape = tuple(
            int(value) for value in arrays["global_features"]
        )
        self.roi_shape = roi_feature_shape(contract)
        self.feature_amp_enabled = bool(amp and device.type == "cuda")


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


def _selected_rows(args: argparse.Namespace) -> list[CacheVideo]:
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        CacheVideo(
            path=str(row["path"]),
            domain=str(row["domain"]),
            split=str(row["split"]),
            label=bool(int(row["label"])),
            label_name=str(row["label_name"]),
            video_path=Path(row["video_path"]),
        )
        for row in rows
        if row["domain"] == args.domain and row["split"] == args.split
    ]
    if not selected:
        raise ValueError("No selected videos matched domain/split")
    return sorted(selected, key=lambda row: row.path.casefold())


def _artifact_path(root: Path, sample_id: str, suffix: str) -> Path:
    relative = Path(*sample_id.split("/"))
    return root / relative.parent / f"{relative.name}{suffix}"


def _boxes_path(args: argparse.Namespace, record: CacheVideo) -> Path:
    root = args.boxes_root / "actor_boxes"
    current = _artifact_path(root, record.path, ".jsonl")
    if current.is_file() or record.domain != "fall_video_dataset":
        return current
    # Older Fall Video Dataset pseudo-label scripts omitted the domain prefix.
    legacy_id = record.path.removeprefix("fall_video_dataset/")
    return _artifact_path(root, legacy_id, ".jsonl")


def _sampled_frames(
    capture: cv2.VideoCapture,
    source_indices: Sequence[int],
    boxes: Mapping[int, tuple[float, float, float, float]],
) -> Iterator[FrameItem]:
    positions: dict[int, list[int]] = {}
    for sample_index, source_index in enumerate(source_indices):
        positions.setdefault(int(source_index), []).append(sample_index)
    maximum = max(positions)
    for source_index in range(maximum + 1):
        ok, frame = capture.read()
        if not ok:
            break
        for sample_index in positions.get(source_index, ()):
            yield sample_index, frame, boxes.get(source_index)


def _history(end_index: int, offsets: Sequence[int]) -> list[int]:
    return [
        end_index - int(offset) if end_index >= int(offset) else -1
        for offset in offsets
    ]


def _cauca_targets(
    record: CaucaVideo,
    source_indices: Sequence[int],
) -> list[bool | None]:
    return [record.frames[index].target for index in source_indices]


def _records(
    record: CacheVideo,
    *,
    source_indices: Sequence[int],
    valid_endpoints: Sequence[int],
    history_offsets: Sequence[int],
    effective_fps: float,
    cauca_record: CaucaVideo | None,
) -> list[dict[str, object]]:
    targets: list[bool | None]
    if cauca_record is None:
        targets = [None] * len(source_indices)
    else:
        targets = _cauca_targets(cauca_record, source_indices)
    rows: list[dict[str, object]] = []
    previous_event = False
    for window_index, end_index in enumerate(valid_endpoints):
        if record.domain == "fall_video_dataset":
            event_target = 0.0
            event_weight = 0.0 if record.label else 1.0
            action_weight = 0.0
            control_weight = 0.0
            action_label = 9
            stage_label = 0
            onset_target = 0.0
            weak_video_label = True
        else:
            target = targets[end_index]
            event_target = float(bool(target))
            event_weight = float(target is not None)
            stage_label = int(bool(target))
            if target:
                action_label = 1
                action_weight = 1.0
            elif record.label:
                action_label = 9
                action_weight = 0.0
            else:
                action_label = ACTION_IDS.get(record.label_name, 9)
                action_weight = float(record.label_name in ACTION_IDS)
            control_weight = action_weight
            onset_target = float(bool(target) and not previous_event)
            previous_event = bool(target)
            weak_video_label = False
        rows.append(
            {
                "path": record.path,
                "domain": record.domain,
                "video_has_fall_event": record.label,
                "weak_video_label": weak_video_label,
                "event_target": event_target,
                "window_index": window_index,
                "end_frame": end_index,
                "end_time_s": end_index / effective_fps,
                "window_action_label": action_label,
                "action_label": action_label,
                "stage_label": stage_label,
                "onset_target": onset_target,
                "event_loss_weight": event_weight,
                "action_loss_weight": action_weight,
                "control_loss_weight": control_weight,
                "history_frame_indices": _history(end_index, history_offsets),
            }
        )
    return rows


def _contract_metadata(
    contract: Mapping[str, object],
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    keys = (
        "history_offsets",
        "visual_offsets",
        "visual_input_size",
        "geometry_input_size",
        "effective_fps",
        "weights_sha256",
        "feature_adapter",
    )
    return {
        "split": args.split,
        "domain": args.domain,
        **{key: contract[key] for key in keys},
        "data_sources": {
            "selection_manifest": str(args.manifest),
            "boxes_root": str(args.boxes_root),
        },
    }


def run_extraction(args: argparse.Namespace) -> Path:
    if args.feature_batch_size <= 0 or args.log_interval <= 0:
        raise ValueError("Batch/log settings must be positive")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    contract_path = args.contract_cache / "manifest.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if _sha256(args.weights) != contract["weights_sha256"]:
        raise ValueError("YOLO weights do not match the feature contract")
    records = _selected_rows(args)
    cauca_by_path = {
        f"caucafall/{record.sample_id}": record
        for record in discover_videos(args.cauca_root)
    } if args.domain == "caucafall" else {}
    device = _device(args.device)
    adapter_spec = contract["feature_adapter"]
    adapter = UltralyticsYOLOFeatureAdapter(
        args.weights,
        global_levels=tuple(adapter_spec["global_levels"]),
        roi_levels=tuple(adapter_spec["roi_levels"]),
        roi_output_size=int(adapter_spec["roi_output_size"]),
        roi_context_scale=float(adapter_spec["roi_context_scale"]),
    ).to(device)
    adapter.eval()
    runtime = _FeatureRuntime(
        adapter=adapter,
        contract=contract,
        device=device,
        amp=args.amp,
    )
    writer = FrozenFeatureCacheWriter(
        args.output_root,
        metadata=_contract_metadata(contract, args=args),
    )
    history_offsets = tuple(int(value) for value in contract["history_offsets"])
    effective_fps = float(contract["effective_fps"])
    started = time.perf_counter()
    written = 0
    skipped = 0
    try:
        for index, record in enumerate(records, start=1):
            capture = cv2.VideoCapture(str(record.video_path))
            if not capture.isOpened():
                skipped += 1
                continue
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            boxes_path = _boxes_path(args, record)
            if min(frame_count, source_fps, width, height) <= 0 or not boxes_path.is_file():
                capture.release()
                skipped += 1
                continue
            boxes = read_actor_box_sequence(
                boxes_path, frame_count=frame_count
            ).boxes
            source_indices = resampled_source_indices(
                frame_count, source_fps, effective_fps
            )
            try:
                sequence = extract_external_sequence(
                    runtime,
                    _sampled_frames(capture, source_indices, boxes),
                    sample_count=len(source_indices),
                    feature_batch_size=args.feature_batch_size,
                )
            finally:
                capture.release()
            valid_endpoints = torch.nonzero(
                sequence.decoded_mask.gt(0)
                & sequence.actor_valid_mask.gt(0),
                as_tuple=False,
            ).flatten().tolist()
            cauca_record = cauca_by_path.get(record.path)
            if cauca_record is not None:
                targets = _cauca_targets(cauca_record, source_indices)
                valid_endpoints = [
                    value for value in valid_endpoints if targets[value] is not None
                ]
            if not valid_endpoints:
                skipped += 1
                continue
            rows = _records(
                record,
                source_indices=source_indices,
                valid_endpoints=valid_endpoints,
                history_offsets=history_offsets,
                effective_fps=effective_fps,
                cauca_record=cauca_record,
            )
            frame_indices = np.arange(len(source_indices), dtype=np.int64)
            writer.write_video(
                record.path,
                frame_indices=frame_indices,
                visual_frame_indices=frame_indices,
                actor_boxes=sequence.geometry_boxes.numpy(),
                actor_valid_mask=sequence.actor_valid_mask.numpy(),
                visual_actor_boxes=sequence.visual_boxes.numpy(),
                global_features=sequence.global_features.numpy(),
                roi_features=sequence.roi_features.numpy(),
                source_size=(height, width),
                records=rows,
            )
            written += 1
            if index % args.log_interval == 0 or index == len(records):
                elapsed = time.perf_counter() - started
                print(
                    f"videos={index}/{len(records)} written={written} "
                    f"skipped={skipped} videos_per_s={index / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        manifest = writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        adapter.close()
    print(f"feature_cache={manifest}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/multidomain_cauca60/split_manifest.csv"),
    )
    parser.add_argument(
        "--domain",
        choices=("fall_video_dataset", "caucafall"),
        required=True,
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--boxes-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--contract-cache",
        type=Path,
        default=Path("artifacts/frozen_features/yolo26x/train"),
    )
    parser.add_argument("--weights", type=Path, default=Path("yolo26x.pt"))
    parser.add_argument(
        "--cauca-root",
        type=Path,
        default=Path("Dataset/caucafall/Dataset CAUCAFall/CAUCAFall"),
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def main() -> None:
    run_extraction(build_parser().parse_args())


if __name__ == "__main__":
    main()
