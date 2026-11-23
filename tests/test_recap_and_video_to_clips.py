import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.modes import get_clip_mode, is_compilation, is_recap, is_video_to_clips
from core.models import Segment
from video.video_to_clips import (
    ActiveSegment,
    VideoPart,
    find_active_segments,
    find_dead_intervals,
    find_snap_boundary,
    partition_active_stream,
)


def test_uniform_mode_mutual_exclusivity():
    # Direct clip_mode overrides
    assert get_clip_mode({"clip_mode": "video_to_clips"}) == "video_to_clips"
    assert is_video_to_clips({"clip_mode": "video_to_clips"}) is True
    assert is_compilation({"clip_mode": "video_to_clips"}) is False
    assert is_recap({"clip_mode": "video_to_clips"}) is False

    assert get_clip_mode({"clip_mode": "compilation"}) == "compilation"
    assert is_compilation({"clip_mode": "compilation"}) is True
    assert is_video_to_clips({"clip_mode": "compilation"}) is False

    assert get_clip_mode({"clip_mode": "recap"}) == "recap"
    assert is_recap({"clip_mode": "recap"}) is True
    assert is_video_to_clips({"clip_mode": "recap"}) is False

    assert get_clip_mode({"clip_mode": "standard"}) == "standard"
    assert is_compilation({"clip_mode": "standard"}) is False
    assert is_video_to_clips({"clip_mode": "standard"}) is False

    # Backward compatibility with boolean flags
    assert get_clip_mode({"video_to_clips": True}) == "video_to_clips"
    assert get_clip_mode({"compilation": True}) == "compilation"
    assert get_clip_mode({"recap": True}) == "recap"
    assert get_clip_mode({}) == "standard"


def test_find_dead_intervals_and_active_segments():
    # Simulate 60 seconds video:
    # 0s - 15s: Active movement & audio
    # 15s - 25s: Frozen frame (motion = 0.1, duration 10s > 4.5s) -> DEAD
    # 25s - 35s: Active DIY / repair / gameplay
    # 35s - 38s: Short pause (motion = 0.2, duration 3s < 4.5s) -> NOT DEAD (kept)
    # 38s - 45s: Active
    # 45s - 55s: Static menu / silence (motion = 1.0, loudness = 0.001, duration 10s > 4.5s) -> DEAD
    # 55s - 60s: Active
    duration = 60.0
    motion = np.full(60, 8.0, dtype=np.float32)
    loudness = np.full(60, 0.05, dtype=np.float32)

    # Dead block 1: frozen frame 15-25s
    motion[15:25] = 0.1

    # Short block 35-38s (only 3s, below 4.5s threshold)
    motion[35:38] = 0.2

    # Dead block 2: static menu + silence 45-55s
    motion[45:55] = 1.0
    loudness[45:55] = 0.001

    visual_raw = {"motion": motion}
    audio_raw = {"loudness": loudness}

    dead = find_dead_intervals(duration, visual_raw, audio_raw, min_dead_duration=4.5)
    assert len(dead) == 2, f"Expected 2 dead blocks, got {dead}"
    assert dead[0] == (15.0, 25.0)
    assert dead[1] == (45.0, 55.0)

    # Active segments with 0.5s safety buffer around dead cuts
    active = find_active_segments(
        duration,
        visual_raw,
        audio_raw,
        min_dead_duration=4.5,
        buffer_seconds=0.5,
        merge_gap=3.0,
    )
    # First dead period 15-25s cut: leaves buffer up to 15.5s, resumes at 24.5s
    # Short pause 35-38s was kept, so 24.5s to 45.5s is one continuous active segment
    # Second dead period 45-55s cut: resumes at 54.5s to 60s
    assert len(active) == 3, f"Expected 3 active segments, got {len(active)}: {active}"
    assert abs(active[0].end - 15.5) <= 0.2
    assert abs(active[1].start - 24.5) <= 0.2
    assert abs(active[1].end - 45.5) <= 0.2
    assert abs(active[2].start - 54.5) <= 0.2


def test_elastic_snap_boundary_and_partitioning():
    # Simulate a long video with active segments totaling 400s:
    # Seg 1: 0s - 200s (200s)
    # Seg 2: 210s - 410s (200s)
    active = [ActiveSegment(0.0, 200.0), ActiveSegment(210.0, 410.0)]

    # Transcript with sentences ending at 172.0s ("...and that is how to fix the engine.")
    # and at 188.0s ("Now let's replace the spark plug.")
    transcripts = [
        Segment(160.0, 172.0, "and that is how to fix the engine."),
        Segment(174.0, 188.0, "Now let's replace the spark plug."),
    ]

    parts = partition_active_stream(
        active,
        transcript_segments=transcripts,
        target_part_duration=180.0,
        elastic_slack=15.0,
        min_tail_duration=45.0,
    )

    # 400s total active divided into ~2 parts (~172s and ~228s)
    assert len(parts) >= 2, f"Expected at least 2 parts, got {len(parts)}"
    p1 = parts[0]
    # First part should snap to sentence ending at 172.0s
    assert abs(p1.end - 172.0) <= 1.0, f"Expected snap near 172.0s, got {p1.end}"


def test_no_mini_tail_under_45s():
    # Active duration is 210s total (target 180s):
    # Remainder after 180s is only 30s (< 45s min_tail_duration).
    # Must NOT create a 30s mini part! Instead, merge everything into 1 part of 210s.
    active = [ActiveSegment(0.0, 210.0)]

    parts = partition_active_stream(
        active,
        transcript_segments=[],
        target_part_duration=180.0,
        elastic_slack=15.0,
        min_tail_duration=45.0,
    )

    assert len(parts) == 1, f"Expected 1 part due to tail <45s merge, got {len(parts)}"
    assert abs(parts[0].total_duration - 210.0) <= 0.1


def test_parse_custom_chapters_and_snapping():
    from video.video_to_clips import parse_custom_chapters

    text = """
    00:00 Intro
    01:45 Step 1: Gathering Resources
    03:10 Step 2: Crafting
    01:15:30 Boss Battle
    """
    chapters = parse_custom_chapters(text)
    assert len(chapters) == 4
    assert chapters[0] == (0.0, "Intro")
    assert chapters[1] == (105.0, "Step 1: Gathering Resources")
    assert chapters[2] == (190.0, "Step 2: Crafting")
    assert chapters[3] == (4530.0, "Boss Battle")

    # Test snapping to chapter timestamp
    # Search window [100.0, 115.0] contains chapter at 105.0s
    snap = find_snap_boundary(100.0, 115.0, chapter_timestamps=[105.0, 190.0])
    assert snap == 105.0

    # Test partitioning with custom chapters
    active = [ActiveSegment(0.0, 400.0)]
    parts = partition_active_stream(
        active,
        target_part_duration=180.0,
        elastic_slack=15.0,
        custom_chapters=text,
    )
    assert len(parts) >= 2
    # First part or second part picks up chapter title
    assert parts[0].chapter_title in ("Intro", "Step 1: Gathering Resources", "Step 2: Crafting")


if __name__ == "__main__":
    test_uniform_mode_mutual_exclusivity()
    test_find_dead_intervals_and_active_segments()
    test_elastic_snap_boundary_and_partitioning()
    test_no_mini_tail_under_45s()
    test_parse_custom_chapters_and_snapping()
    print("ALL RONDE 1 VIDEO TO CLIPS TESTS PASSED SUCCESSFULLY!")
