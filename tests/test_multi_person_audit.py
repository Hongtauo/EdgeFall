from pathlib import Path

from edgefall_st.pseudo.multi_person_audit import _write_html, analyze_video


def _detection(track_id: int, box: list[float]) -> dict[str, object]:
    return {
        "track_id": track_id,
        "bbox_xyxy_norm": box,
        "confidence": 0.9,
        "area_ratio": (box[2] - box[0]) * (box[3] - box[1]),
    }


def test_analyze_video_finds_consistent_second_person(tmp_path: Path) -> None:
    frames = []
    primary = []
    for frame_index in range(10):
        actor_box = [0.1, 0.1, 0.35, 0.9]
        second_box = [0.6, 0.15, 0.85, 0.9]
        frames.append(
            {
                "path": "fall/example",
                "frame_index": frame_index,
                "timestamp_s": frame_index / 25.0,
                "action_label": 1,
                "detections": [
                    _detection(1, actor_box),
                    _detection(2, second_box),
                ],
            }
        )
        primary.append(
            {
                "actor_detection_index": 0,
                "actor_source_track_id": 1,
                "actor_bbox_xyxy_norm": actor_box,
            }
        )

    record, secondary = analyze_video(
        path="fall/example",
        track_frames=frames,
        primary_targets=primary,
        dataset_root=tmp_path,
        min_coexist_frames=8,
        distinct_iou_threshold=0.65,
    )

    assert record is not None
    assert record.auto_group == "spatially_distinct"
    assert record.primary_track_ids == "1"
    assert record.secondary_track_ids == "2"
    assert record.coexist_frames == 10
    assert record.distinct_coexist_frames == 10
    assert all(row["actor_source_track_id"] == 2 for row in secondary)


def test_analyze_video_ignores_brief_second_detection(tmp_path: Path) -> None:
    frames = []
    primary = []
    for frame_index in range(10):
        actor_box = [0.1, 0.1, 0.35, 0.9]
        detections = [_detection(1, actor_box)]
        if frame_index < 3:
            detections.append(_detection(2, [0.6, 0.15, 0.85, 0.9]))
        frames.append(
            {
                "path": "fall/example",
                "frame_index": frame_index,
                "timestamp_s": frame_index / 25.0,
                "action_label": 1,
                "detections": detections,
            }
        )
        primary.append(
            {
                "actor_detection_index": 0,
                "actor_source_track_id": 1,
                "actor_bbox_xyxy_norm": actor_box,
            }
        )

    record, _secondary = analyze_video(
        path="fall/example",
        track_frames=frames,
        primary_targets=primary,
        dataset_root=tmp_path,
        min_coexist_frames=8,
        distinct_iou_threshold=0.65,
    )

    assert record is None


def test_write_html_keeps_javascript_escapes(tmp_path: Path) -> None:
    output = tmp_path / "index.html"

    _write_html(output, (), video_directory="videos_h264")

    document = output.read_text(encoding="utf-8")
    assert 'const videoDirectory="videos_h264";' in document
    assert 'rows.join("\\r\\n")' in document
    assert "html,body{height:100%;overflow:hidden}" in document
    assert "#list{overflow:auto;overscroll-behavior:contain" in document
    assert "main{padding:16px 20px;overflow:auto;overscroll-behavior:contain" in document
