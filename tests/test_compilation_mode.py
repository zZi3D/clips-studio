import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.fusion import bundle_compilation_highlights, frame_compilation_window
from core.modes import is_compilation


def test_bundle_compilation_highlights():
    # 3 highlights within 18s of each other (10.0s, 22.0s, 35.0s)
    # Then a 30s gap of quiet gameplay -> highlight at 65.0s
    events = [
        (10.0, "VISUAL: Car flips mid-air and explodes"),
        (22.0, "AUDIO spike (shouting/laughter/cheering likely)"),
        (35.0, "VISUAL: Helicopter crash stunt"),
        (65.0, "VISUAL: Funny ragdoll glitch"),
    ]

    bundled = bundle_compilation_highlights(
        events,
        min_duration=25.0,
        max_duration=90.0,
        merge_gap_seconds=18.0,
        pre_climax_seconds=4.5,
        post_climax_seconds=6.5,
        video_end=120.0,
    )

    # Must produce exactly 2 coherent clips (first 3 bundled together, 4th separate)
    assert len(bundled) == 2, f"Expected 2 bundled clips, got {len(bundled)}"

    clip1_start, clip1_end, cluster1 = bundled[0]
    assert len(cluster1) == 3, f"Expected 3 events in cluster 1, got {len(cluster1)}"
    # First highlight was at 10.0s -> start must be at least 4.5s earlier (<= 5.5s)
    assert clip1_start <= 10.0 - 4.5, f"Expected start <= 5.5, got {clip1_start}"
    # Last highlight in bundle was at 35.0s -> end must be at least 6.5s later (>= 41.5s)
    assert clip1_end >= 35.0 + 6.5, f"Expected end >= 41.5, got {clip1_end}"
    # Duration must be within bounds [25.0, 90.0]
    assert 25.0 <= (clip1_end - clip1_start) <= 90.0, f"Clip 1 duration {clip1_end - clip1_start} out of bounds"

    clip2_start, clip2_end, cluster2 = bundled[1]
    assert len(cluster2) == 1, f"Expected 1 event in cluster 2, got {len(cluster2)}"
    # Highlight at 65.0s -> must have at least 4.5s pre-climax and 6.5s post-climax
    assert clip2_start <= 65.0 - 4.5, f"Expected start <= 60.5, got {clip2_start}"
    assert clip2_end >= 65.0 + 6.5, f"Expected end >= 71.5, got {clip2_end}"
    # Must enforce min_duration 25.0s
    assert clip2_end - clip2_start >= 25.0, f"Expected duration >= 25.0, got {clip2_end - clip2_start}"


def test_frame_compilation_window_margins_and_no_aggressive_cut():
    # Scene cuts exist in the video near the action (e.g. at 28.0s and 37.0s)
    cuts = [28.0, 37.0, 50.0]

    # Action from 30.0s to 35.0s
    # pre_climax = 4.5s, post_climax = 6.5s, min_duration = 25.0s
    s, e = frame_compilation_window(
        30.0, 35.0,
        cut_times=cuts,
        min_duration=25.0,
        max_duration=90.0,
        pre_climax_seconds=4.5,
        post_climax_seconds=6.5,
        video_end=150.0,
    )

    # 1. Start must be at least 4.5s before action (<= 25.5s)
    assert s <= 30.0 - 4.5, f"Expected start <= 25.5, got {s}"
    # 2. End must be at least 6.5s after action (>= 41.5s) -- cuts_raw at 37.0s must NOT truncate!
    assert e >= 35.0 + 6.5, f"Expected end >= 41.5 (not truncated by cut at 37s), got {e}"
    # 3. Duration must be at least min_duration (25.0s)
    assert e - s >= 25.0, f"Expected duration >= 25.0, got {e - s}"


def test_is_compilation():
    assert is_compilation({"compilation": True}) is True
    assert is_compilation({"clips": {"compilation": True}}) is True
    assert is_compilation({"clips": {"gaming": True}}) is False
    assert is_compilation({}) is False


if __name__ == "__main__":
    test_bundle_compilation_highlights()
    test_frame_compilation_window_margins_and_no_aggressive_cut()
    test_is_compilation()
    print("ALL COMPILATION MODE TESTS PASSED!")
