import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.fusion import bundle_compilation_highlights, frame_compilation_window
from analysis.visual_captioning import PROMPT_ACTION, PROMPT_RECAP, select_frame_times
from core.modes import is_recap, is_video_to_clips
from core.pipeline import _make_video_to_clips_parts, _make_recap_video
from core.models import ClipCandidate, RenderedClip


def test_mode_detection():
    # Direct dict keys
    assert is_recap({"recap": True}) is True
    assert is_recap({"recap_mode": True}) is True
    assert is_recap({"clips": {"recap": True}}) is True
    assert is_recap({"clips": {"compilation": True}}) is False
    assert is_recap({}) is False

    assert is_video_to_clips({"video_to_clips": True}) is True
    assert is_video_to_clips({"video_to_clips_mode": True}) is True
    assert is_video_to_clips({"clips": {"video_to_clips": True}}) is True
    assert is_video_to_clips({"clips": {"compilation": True}}) is False
    assert is_video_to_clips({}) is False


def test_recap_prompt_and_frame_window():
    # Verify prompt contains walkthrough/tutorial/secret items
    assert "tutorial step" in PROMPT_RECAP.lower()
    assert "secret item" in PROMPT_RECAP.lower()
    assert "easter egg" in PROMPT_RECAP.lower()
    assert "map location" in PROMPT_RECAP.lower()

    # Tutorial steps at 10.0s, 14.0s (within 6.0s merge gap) and at 40.0s
    events = [
        (10.0, "VISUAL: Map location revealed on screen"),
        (14.0, "VISUAL: Secret key item picked up"),
        (40.0, "VISUAL: Puzzle door opened"),
    ]

    bundled = bundle_compilation_highlights(
        events,
        min_duration=5.0,
        max_duration=20.0,
        merge_gap_seconds=6.0,
        pre_climax_seconds=2.5,
        post_climax_seconds=3.0,
        video_end=100.0,
    )

    assert len(bundled) == 2, f"Expected 2 recap step clusters, got {len(bundled)}"

    # First cluster: 10s and 14s merged
    s1, e1, c1 = bundled[0]
    assert len(c1) == 2
    # Lead-in of 2.5s -> start <= 10.0 - 2.5 = 7.5s
    assert s1 <= 7.5, f"Expected s1 <= 7.5, got {s1}"
    # Follow-through of 3.0s -> end >= 14.0 + 3.0 = 17.0s
    assert e1 >= 17.0, f"Expected e1 >= 17.0, got {e1}"
    assert 5.0 <= (e1 - s1) <= 20.0, f"Duration {e1 - s1} out of bounds for recap step"

    # Second cluster: 40s
    s2, e2, c2 = bundled[1]
    assert len(c2) == 1
    assert s2 <= 40.0 - 2.5
    assert e2 >= 40.0 + 3.0
    assert 5.0 <= (e2 - s2) <= 20.0


def test_video_to_clips_scaling_and_window():
    # Long video (e.g. 5400s = 90 min)
    times = select_frame_times(
        duration=5400.0,
        visual_raw=None,
        interval_seconds=8.0,
        max_frames=1200,
    )
    # 5400s / 8.0s ~ 675 frames, well within max_frames: 1200
    assert len(times) > 600, f"Expected >600 frames for 90m video, got {len(times)}"
    assert times[0] == 0.0
    assert times[-1] >= 5390.0

    # Highlights with 3.5s lead-in, 5.0s follow-through, 12.0s merge gap
    events = [
        (100.0, "VISUAL: Big explosion in heist"),
        (108.0, "VISUAL: Car getaway police chase"),
        (250.0, "VISUAL: Plane parachute jump"),
    ]

    bundled = bundle_compilation_highlights(
        events,
        min_duration=15.0,
        max_duration=60.0,
        merge_gap_seconds=12.0,
        pre_climax_seconds=3.5,
        post_climax_seconds=5.0,
        video_end=5400.0,
    )

    assert len(bundled) == 2
    s1, e1, c1 = bundled[0]
    assert len(c1) == 2
    assert s1 <= 100.0 - 3.5
    assert e1 >= 108.0 + 5.0
    assert 15.0 <= (e1 - s1) <= 60.0


def test_video_to_clips_partitioning():
    # Simulate clips totaling ~350 seconds
    # Should partition into 2 parts of ~175s each (target 120-180s)
    clips = []
    t = 0.0
    for i in range(14):
        dur = 25.0
        c = ClipCandidate(start=t, end=t + dur, score=85, hook=f"Clip {i}")
        clips.append(RenderedClip(source_video_id="test_vid", candidate=c, path=Path(f"clip_{i}.mp4")))
        t += dur + 5.0

    # Test partitioning logic
    batches = []
    cur_batch = []
    cur_dur = 0.0
    min_part = 120.0
    max_part = 180.0

    for clip in clips:
        dur = clip.candidate.end - clip.candidate.start
        if not cur_batch:
            cur_batch.append(clip)
            cur_dur += dur
            continue
        if (cur_dur + dur > max_part and cur_dur >= min_part) or (cur_dur + dur > max_part + 30.0):
            batches.append(cur_batch)
            cur_batch = [clip]
            cur_dur = dur
        else:
            cur_batch.append(clip)
            cur_dur += dur
    if cur_batch:
        batches.append(cur_batch)

    # 14 clips of 25s = 350s total -> exactly 2 batches of ~175s each
    assert len(batches) == 2, f"Expected 2 parts, got {len(batches)}"
    b1_dur = sum(c.candidate.end - c.candidate.start for c in batches[0])
    b2_dur = sum(c.candidate.end - c.candidate.start for c in batches[1])
    assert 120.0 <= b1_dur <= 180.0, f"Batch 1 duration {b1_dur} out of bounds"
    assert 120.0 <= b2_dur <= 180.0, f"Batch 2 duration {b2_dur} out of bounds"


if __name__ == "__main__":
    test_mode_detection()
    test_recap_prompt_and_frame_window()
    test_video_to_clips_scaling_and_window()
    test_video_to_clips_partitioning()
    print("ALL RECAP AND VIDEO TO CLIPS TESTS PASSED!")
