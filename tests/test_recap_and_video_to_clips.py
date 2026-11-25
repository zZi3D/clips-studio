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


def test_youtube_markdown_chapters_and_strict_180s_binpacking():
    from video.video_to_clips import parse_custom_chapters

    # Exact raw text from user copy-pasted from YouTube
    raw_text = """
    [0:00](https://www.youtube.com/watch?v=r3oAEokOT5w) Intro [0:29](https://www.youtube.com/watch?v=r3oAEokOT5w&t=29s) Mistake #1 - Spending Money Too Early [2:04](https://www.youtube.com/watch?v=r3oAEokOT5w&t=124s) Mistake #2 - Missing FREE 50K [3:05](https://www.youtube.com/watch?v=r3oAEokOT5w&t=185s) Mistake #3 - Not Upgrading Your Weapons [4:04](https://www.youtube.com/watch?v=r3oAEokOT5w&t=244s) Mistake #4 - Forgetting to Save Your Game [5:00](https://www.youtube.com/watch?v=r3oAEokOT5w&t=300s) Mistake #5 – Driving EVERYWHERE [5:51](https://www.youtube.com/watch?v=r3oAEokOT5w&t=351s) Mistake #6 - Upgrading the Wrong Cars [6:21](https://www.youtube.com/watch?v=r3oAEokOT5w&t=381s) Mistake #7 - Trusting Autosave [7:18](https://www.youtube.com/watch?v=r3oAEokOT5w&t=438s) Mistake #8 – MOST IMPORTANT MISTAKE [9:38](https://www.youtube.com/watch?v=r3oAEokOT5w&t=578s) Mistake #9 - Wasting Money on THIS [10:53](https://www.youtube.com/watch?v=r3oAEokOT5w&t=653s) Mistake #10 - Underestimating Missions [12:04](https://www.youtube.com/watch?v=r3oAEokOT5w&t=724s) Mistake #11 - Robbing Stores the Wrong Way [12:48](https://www.youtube.com/watch?v=r3oAEokOT5w&t=768s) Mistake #12 - Ignoring Scuba Diving [13:14](https://www.youtube.com/watch?v=r3oAEokOT5w&t=794s) Mistake #13 - Missing Criminal Opportunities [13:49](https://www.youtube.com/watch?v=r3oAEokOT5w&t=829s&pp=0gcJCWMAwfN6Pr3D) Mistake #14 - Believing Everything Online [15:12](https://www.youtube.com/watch?v=r3oAEokOT5w&t=912s) Mistake #15 - Rushing Through GTA 6
    """
    chapters = parse_custom_chapters(raw_text)
    assert len(chapters) == 16
    assert chapters[0][1] == "Intro"
    assert "Mistake #1" in chapters[1][1]
    assert "Mistake #8" in chapters[8][1]

    # Partition full video (1000s active gameplay)
    active = [ActiveSegment(0.0, 1000.0)]
    parts = partition_active_stream(active, custom_chapters=raw_text)

    # 1. Every single part must strictly obey YouTube Shorts <= 180.0s (3 min) limit
    for p in parts:
        assert p.total_duration <= 180.0, f"Part {p.part_number} ({p.total_duration}s) exceeded 180s YouTube Shorts limit!"

    # 2. Mistake #8 (140s alone) must have its own part and not bleed into Mistake #7 or #9
    p8 = [p for p in parts if "Mistake #8" in p.chapter_title]
    assert len(p8) == 1
    assert p8[0].total_duration == 140.0
    assert "Mistake #7" not in p8[0].chapter_title
    assert "Mistake #9" not in p8[0].chapter_title


if __name__ == "__main__":
    test_uniform_mode_mutual_exclusivity()
    test_find_dead_intervals_and_active_segments()
    test_elastic_snap_boundary_and_partitioning()
    test_no_mini_tail_under_45s()
    test_parse_custom_chapters_and_snapping()
    test_youtube_markdown_chapters_and_strict_180s_binpacking()
    print("ALL RONDE 1 VIDEO TO CLIPS TESTS PASSED SUCCESSFULLY!")
