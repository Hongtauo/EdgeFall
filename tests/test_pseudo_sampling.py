from __future__ import annotations

from collections import Counter

from edgefall_st.pseudo.sampling import PilotVideo, select_pilot_videos
from edgefall_st.pseudo.quality import VideoTrackQuality, summarize_quality


def make_candidates() -> list[PilotVideo]:
    videos = []
    for index in range(320):
        labels = (index % 16,)
        videos.append(
            PilotVideo(
                path=f"folder_{index % 10}/video_{index:03d}",
                split="train",
                labels=labels,
                metadata={
                    "environment_category": f"env_{index % 2}",
                    "camera_shot": f"shot_{index % 3}",
                    "speed": f"speed_{index % 4}",
                    "camera_elevation": f"elevation_{index % 3}",
                    "camera_azimuth": f"azimuth_{index % 4}",
                    "camera_distance": f"distance_{index % 3}",
                    "age_group": f"age_{index % 6}",
                    "bmi_band": f"bmi_{index % 3}",
                },
            )
        )
    return videos


def test_pilot_selection_is_deterministic_and_covers_labels() -> None:
    candidates = make_candidates()
    first = select_pilot_videos(candidates, sample_size=160, seed=18)
    second = select_pilot_videos(candidates, sample_size=160, seed=18)

    assert [video.path for video in first] == [video.path for video in second]
    assert len({video.path for video in first}) == 160
    label_counts = Counter(label for video in first for label in video.labels)
    assert all(label_counts[label] >= 10 for label in range(16))


def test_full_selection_returns_every_video_deterministically() -> None:
    candidates = make_candidates()

    first = select_pilot_videos(
        candidates,
        sample_size=len(candidates),
        seed=18,
    )
    second = select_pilot_videos(
        list(reversed(candidates)),
        sample_size=len(candidates),
        seed=18,
    )

    assert [video.path for video in first] == [video.path for video in second]
    assert {video.path for video in first} == {
        video.path for video in candidates
    }


def test_quality_summary_flags_low_coverage() -> None:
    rows = [
        VideoTrackQuality(
            path="fall/a",
            expected_frames=100,
            processed_frames=100,
            fps=30.0,
            width=640,
            height=360,
            frames_with_person=100,
            multi_detection_frames=0,
            untracked_detection_frames=0,
            unique_track_ids=1,
            dominant_track_frames=100,
            mean_person_confidence=0.9,
            median_bbox_area_ratio=0.2,
            elapsed_s=1.0,
        ),
        VideoTrackQuality(
            path="fallen/b",
            expected_frames=100,
            processed_frames=100,
            fps=30.0,
            width=640,
            height=360,
            frames_with_person=80,
            multi_detection_frames=0,
            untracked_detection_frames=2,
            unique_track_ids=2,
            dominant_track_frames=70,
            mean_person_confidence=0.7,
            median_bbox_area_ratio=0.1,
            elapsed_s=1.2,
        ),
    ]

    summary = summarize_quality(rows)

    assert summary["processed_videos"] == 2
    assert summary["mean_detection_coverage"] == 0.9
    assert summary["mean_dominant_track_coverage"] == 0.85
    assert summary["videos_below_95pct_detection"] == ["fallen/b"]
    assert summary["videos_below_90pct_dominant"] == ["fallen/b"]
    assert summary["videos_with_multiple_track_ids"] == ["fallen/b"]
    assert summary["source_action_summary"]["fall"]["videos"] == 1
    assert (
        summary["source_action_summary"]["fallen"][
            "videos_below_90pct_dominant"
        ]
        == 1
    )
