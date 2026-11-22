"""Domain-independent Video to Clips (2-3 min) engine.

Designed for long recordings (90 to 300+ minutes) across any domain
(car repair, DIY tutorials, tech how-tos, coding, longform gameplay):
1. Dead-time detection (>4.5s) using lightweight visual motion and audio RMS:
   - Frozen frames / static screens (loading screens, pause menus)
   - Low motion with silence (AFK / inactive footage)
   - 0.5s safety buffer before and after cuts; merges active gaps <= 3.0s.
2. Snap-to-boundary natural partitioner:
   - Groups active footage into sequential parts of 120-180s (part_01, part_02, ...).
   - Elastic extension (+10 to +15s slack) snaps to sentence boundaries (transcript),
     dips in motion/audio, or scene cuts.
   - Prevents tiny tail fragments: if remainder is < 45s, merges it into the prior part.
3. FFmpeg export with micro audio-crossfades (40ms):
   - Uses script/listing files to avoid Windows command-line length limits.
   - Re-encodes cleanly at cutpoints with 40ms micro audio-fade transitions.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from core import cancel
from core.binaries import ffmpeg
from core.models import ClipCandidate, RenderedClip, Segment
from core.paths import discard
from core.state import StateDB
from video.encoding import concat, concat_entry


@dataclass
class ActiveSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class VideoPart:
    part_number: int
    segments: list[ActiveSegment]
    start: float
    end: float
    total_duration: float

    @property
    def label(self) -> str:
        return f"part_{self.part_number:02d}.mp4"


def find_dead_intervals(
    duration: float,
    visual_raw: dict | None = None,
    audio_raw: dict | None = None,
    min_dead_duration: float = 4.5,
    freeze_motion_threshold: float = 0.6,
    static_motion_threshold: float = 1.8,
    low_motion_threshold: float = 3.0,
    silence_threshold_ratio: float = 0.15,
) -> list[tuple[float, float]]:
    """Detect dead periods (>min_dead_duration) from motion and audio signals.

    A second t is considered dead if:
      a) Image is frozen: motion <= freeze_motion_threshold
      b) Screen is almost static (menus): motion <= static_motion_threshold
         AND no audio burst/spike
      c) AFK / idle: motion <= low_motion_threshold AND audio is silent
    """
    duration = max(0.0, float(duration))
    if duration <= 0:
        return []

    n_secs = int(np.ceil(duration))
    motion = (visual_raw or {}).get("motion")
    loudness = (audio_raw or {}).get("loudness")

    if motion is None or len(motion) == 0:
        motion_arr = np.full(n_secs, 10.0, dtype=np.float32)
    else:
        if len(motion) < n_secs:
            pad = np.full(n_secs - len(motion), motion[-1] if len(motion) > 0 else 10.0)
            motion_arr = np.concatenate([motion, pad]).astype(np.float32)
        else:
            motion_arr = np.array(motion[:n_secs], dtype=np.float32)

    if loudness is None or len(loudness) == 0:
        loudness_arr = np.full(n_secs, 0.05, dtype=np.float32)
        silence_thresh = 0.005
    else:
        if len(loudness) < n_secs:
            pad = np.full(n_secs - len(loudness), loudness[-1] if len(loudness) > 0 else 0.05)
            loudness_arr = np.concatenate([loudness, pad]).astype(np.float32)
        else:
            loudness_arr = np.array(loudness[:n_secs], dtype=np.float32)

        valid_loud = loudness_arr[loudness_arr > 1e-5]
        median_loud = float(np.median(valid_loud)) if len(valid_loud) > 0 else 0.05
        silence_thresh = max(0.003, median_loud * silence_threshold_ratio)

    is_dead = np.zeros(n_secs, dtype=bool)
    for sec in range(n_secs):
        m = motion_arr[sec]
        l = loudness_arr[sec]

        # Condition a: Frozen frame / loading screen
        if m <= freeze_motion_threshold:
            is_dead[sec] = True
        # Condition b: Static menu / inventory screen
        elif m <= static_motion_threshold and l <= (silence_thresh * 2.0):
            is_dead[sec] = True
        # Condition c: Low motion with silence (AFK / empty frame)
        elif m <= low_motion_threshold and l <= silence_thresh:
            is_dead[sec] = True

    # Identify contiguous dead blocks >= min_dead_duration
    dead_intervals: list[tuple[float, float]] = []
    in_dead = False
    start_sec = 0

    for sec in range(n_secs):
        if is_dead[sec]:
            if not in_dead:
                in_dead = True
                start_sec = sec
        else:
            if in_dead:
                in_dead = False
                dur = sec - start_sec
                if dur >= min_dead_duration:
                    dead_intervals.append((float(start_sec), float(sec)))

    if in_dead:
        dur = n_secs - start_sec
        if dur >= min_dead_duration:
            dead_intervals.append((float(start_sec), min(float(n_secs), duration)))

    return dead_intervals


def find_active_segments(
    duration: float,
    visual_raw: dict | None = None,
    audio_raw: dict | None = None,
    min_dead_duration: float = 4.5,
    buffer_seconds: float = 0.5,
    merge_gap: float = 3.0,
) -> list[ActiveSegment]:
    """Invert dead intervals into active segments with safety buffers and merging.

    - Retains buffer_seconds (0.5s) before and after each removed dead period.
    - Merges active segments separated by <= merge_gap (3.0s).
    """
    duration = max(0.0, float(duration))
    if duration <= 0:
        return []

    dead_intervals = find_dead_intervals(
        duration,
        visual_raw=visual_raw,
        audio_raw=audio_raw,
        min_dead_duration=min_dead_duration,
    )

    if not dead_intervals:
        return [ActiveSegment(0.0, duration)]

    # Invert dead intervals to active intervals, leaving buffer_seconds buffer
    active_spans: list[tuple[float, float]] = []
    curr = 0.0

    for d_start, d_end in dead_intervals:
        # Give dead pause a buffer: keep active up to d_start + buffer_seconds
        cut_start = min(d_start + buffer_seconds, d_end)
        # Resume active buffer_seconds before d_end
        cut_end = max(d_end - buffer_seconds, cut_start)

        if cut_start > curr + 0.1:
            active_spans.append((curr, cut_start))
        curr = max(curr, cut_end)

    if curr < duration - 0.1:
        active_spans.append((curr, duration))

    if not active_spans:
        return [ActiveSegment(0.0, duration)]

    # Merge active spans separated by <= merge_gap
    merged: list[ActiveSegment] = []
    cur_start, cur_end = active_spans[0]

    for span_start, span_end in active_spans[1:]:
        if span_start - cur_end <= merge_gap:
            cur_end = max(cur_end, span_end)
        else:
            merged.append(ActiveSegment(round(cur_start, 2), round(cur_end, 2)))
            cur_start, cur_end = span_start, span_end

    merged.append(ActiveSegment(round(cur_start, 2), round(cur_end, 2)))
    return [s for s in merged if s.duration > 0.2]


def find_snap_boundary(
    window_start: float,
    window_end: float,
    transcript_segments: Sequence[Segment] | None = None,
    cut_times: Sequence[float] | None = None,
    motion: np.ndarray | None = None,
    loudness: np.ndarray | None = None,
) -> float:
    """Find the best natural split point within [window_start, window_end].

    Priority:
      1. Transcript sentence boundary (end of sentence with '.', '?', '!' or speech gap > 0.5s)
      2. Scene cut point
      3. Valley in combined motion & audio
      4. Fallback: window_start (or midpoint)
    """
    if window_end <= window_start:
        return window_start

    midpoint = (window_start + window_end) / 2.0

    # 1. Transcript sentence boundary
    if transcript_segments:
        best_speech_t = None
        best_speech_dist = 999999.0

        for seg in transcript_segments:
            # End of segment inside search window
            if window_start <= seg.end <= window_end:
                text = (seg.text or "").strip()
                # Punctuation check
                if text.endswith((".", "!", "?")):
                    dist = abs(seg.end - midpoint)
                    if dist < best_speech_dist:
                        best_speech_dist = dist
                        best_speech_t = seg.end

        if best_speech_t is not None:
            return round(best_speech_t, 2)

        # Gap between segments > 0.5s inside window
        for i in range(len(transcript_segments) - 1):
            s1 = transcript_segments[i]
            s2 = transcript_segments[i + 1]
            if s2.start > s1.end + 0.5:
                gap_mid = (s1.end + s2.start) / 2.0
                if window_start <= gap_mid <= window_end:
                    dist = abs(gap_mid - midpoint)
                    if dist < best_speech_dist:
                        best_speech_dist = dist
                        best_speech_t = gap_mid

        if best_speech_t is not None:
            return round(best_speech_t, 2)

    # 2. Scene cuts
    if cut_times:
        valid_cuts = [c for c in cut_times if window_start <= c <= window_end]
        if valid_cuts:
            # Choose cut closest to target midpoint
            best_cut = min(valid_cuts, key=lambda c: abs(c - midpoint))
            return round(best_cut, 2)

    # 3. Motion & Audio valley
    if motion is not None and loudness is not None:
        start_idx = max(0, int(window_start))
        end_idx = min(len(motion), len(loudness), int(np.ceil(window_end)))
        if end_idx > start_idx + 1:
            m_slice = motion[start_idx:end_idx]
            l_slice = loudness[start_idx:end_idx]
            # Combined energy valley
            m_norm = m_slice / (np.max(m_slice) + 1e-5)
            l_norm = l_slice / (np.max(l_slice) + 1e-5)
            combined = m_norm + 1.5 * l_norm
            min_offset = int(np.argmin(combined))
            valley_t = float(start_idx + min_offset)
            if window_start <= valley_t <= window_end:
                return round(valley_t, 2)

    return round(midpoint, 2)


def partition_active_stream(
    active_segments: list[ActiveSegment],
    transcript_segments: Sequence[Segment] | None = None,
    cut_times: Sequence[float] | None = None,
    motion: np.ndarray | None = None,
    loudness: np.ndarray | None = None,
    target_part_duration: float = 180.0,
    elastic_slack: float = 15.0,
    min_tail_duration: float = 45.0,
) -> list[VideoPart]:
    """Partition active footage into sequential VideoParts of 120-180s.

    - Elastic extension: up to target_part_duration + elastic_slack (e.g. 195s)
      to snap to natural speech/scene/energy boundaries.
    - No mini-tail: if remaining active footage is < min_tail_duration (45s),
      merges it into the previous part.
    """
    if not active_segments:
        return []

    total_active = sum(s.duration for s in active_segments)
    if total_active <= (target_part_duration + elastic_slack + min_tail_duration):
        return [
            VideoPart(
                part_number=1,
                segments=active_segments,
                start=active_segments[0].start,
                end=active_segments[-1].end,
                total_duration=round(total_active, 2),
            )
        ]

    # Flatten active segments into linear timeline points
    parts: list[VideoPart] = []
    seg_idx = 0
    seg_offset = active_segments[0].start

    while seg_idx < len(active_segments):
        part_segments: list[ActiveSegment] = []
        accum_dur = 0.0
        part_start_t = seg_offset

        while seg_idx < len(active_segments):
            cur_seg = active_segments[seg_idx]
            avail = cur_seg.end - seg_offset

            # Remaining active duration in the whole video after this point
            remaining_total = sum(s.duration for s in active_segments[seg_idx + 1 :]) + (cur_seg.end - seg_offset)

            # If adding the entire remaining segment keeps us under target_part_duration
            if accum_dur + avail <= target_part_duration:
                part_segments.append(ActiveSegment(seg_offset, cur_seg.end))
                accum_dur += avail
                seg_idx += 1
                if seg_idx < len(active_segments):
                    seg_offset = active_segments[seg_idx].start
                continue

            # We need to split inside this segment or snap near target
            needed = target_part_duration - accum_dur
            ideal_split = seg_offset + needed

            # Search window for elastic snap: [ideal_split - 20s, ideal_split + elastic_slack]
            snap_min = max(seg_offset, ideal_split - 20.0)
            snap_max = min(cur_seg.end, ideal_split + elastic_slack)

            snap_t = find_snap_boundary(
                snap_min,
                snap_max,
                transcript_segments=transcript_segments,
                cut_times=cut_times,
                motion=motion,
                loudness=loudness,
            )

            # Check remainder after snap_t:
            rem_in_seg = cur_seg.end - snap_t
            rem_after_all = rem_in_seg + sum(s.duration for s in active_segments[seg_idx + 1 :])

            # If the remainder would be smaller than min_tail_duration (<45s),
            # swallow the rest into this part!
            if rem_after_all < min_tail_duration:
                # Include the rest of this segment
                part_segments.append(ActiveSegment(seg_offset, cur_seg.end))
                accum_dur += avail
                seg_idx += 1
                # Include all remaining segments if any
                while seg_idx < len(active_segments):
                    rem_seg = active_segments[seg_idx]
                    part_segments.append(rem_seg)
                    accum_dur += rem_seg.duration
                    seg_idx += 1
                break

            # Normal split at snap_t
            part_segments.append(ActiveSegment(seg_offset, snap_t))
            accum_dur += (snap_t - seg_offset)
            seg_offset = snap_t
            break

        if part_segments:
            parts.append(
                VideoPart(
                    part_number=len(parts) + 1,
                    segments=part_segments,
                    start=part_segments[0].start,
                    end=part_segments[-1].end,
                    total_duration=round(accum_dur, 2),
                )
            )

    return parts


def render_video_part(
    video_path: Path,
    part: VideoPart,
    output_path: Path,
    config: dict | None = None,
    crop_filter: str | None = None,
) -> Path:
    """Render a VideoPart into a continuous MP4 file with 40ms micro audio-crossfades.

    Uses an external filter_complex script file to stay safely under
    the Windows 8192-character command-line length limit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / ".v2c_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    script_path = temp_dir / f"{output_path.stem}_filter.txt"
    try:
        # Build filter script for N segments with 40ms audio crossfade / fade
        filter_lines: list[str] = []
        n_segs = len(part.segments)

        for i, seg in enumerate(part.segments):
            dur = seg.duration
            v_label = f"v{i}"
            a_label = f"a{i}"

            # Video trim
            if crop_filter:
                filter_lines.append(
                    f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS,{crop_filter}[{v_label}];"
                )
            else:
                filter_lines.append(
                    f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS[{v_label}];"
                )

            # Audio trim with 40ms (0.04s) micro-fade at head and tail
            fade_d = min(0.04, dur / 3.0)
            st_out = max(0.0, dur - fade_d)
            filter_lines.append(
                f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:ss=0:d={fade_d:.3f},afade=t=out:st={st_out:.3f}:d={fade_d:.3f}[{a_label}];"
            )

        # Concat filter
        v_inputs = "".join(f"[v{i}]" for i in range(n_segs))
        a_inputs = "".join(f"[a{i}]" for i in range(n_segs))
        filter_lines.append(f"{v_inputs}{a_inputs}concat=n={n_segs}:v=1:a=1[v_out][a_out]")

        script_path.write_text("\n".join(filter_lines), encoding="utf-8")

        cmd = [
            ffmpeg(),
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-filter_complex_script",
            str(script_path),
            "-map",
            "[v_out]",
            "-map",
            "[a_out]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg render failed for {output_path.name}: {r.stderr.strip()[-300:]}")

        return output_path
    finally:
        discard(script_path)


def run_video_to_clips_pipeline(
    video_path: Path,
    duration: float,
    visual_raw: dict,
    audio_raw: dict,
    transcript_segments: list[Segment],
    clip_dir: Path,
    config: dict,
    db: StateDB | None = None,
    video_id: str = "",
) -> list[RenderedClip]:
    """Full execution of Video to Clips (2-3 min) mode.

    - Scans full timeline without LLM vision bottleneck.
    - Removes dead moments (>4.5s) with 0.5s safety buffer.
    - Partitions active footage into 120-180s sequential VideoParts.
    - Renders part_01.mp4, part_02.mp4, etc. with 40ms micro audio-crossfades.
    - Registers parts in DB and returns RenderedClip objects.
    """
    v2c_cfg = config.get("video_to_clips_mode", {})
    min_dead = float(v2c_cfg.get("min_dead_duration", 4.5))
    target_dur = float(v2c_cfg.get("target_part_duration", 180.0))
    slack = float(v2c_cfg.get("elastic_slack", 15.0))
    min_tail = float(v2c_cfg.get("min_tail_duration", 45.0))

    print(f"  Video to Clips Mode: scanning full {duration / 60:.1f} min video for active content...")
    t0 = time.monotonic()

    active_segs = find_active_segments(
        duration,
        visual_raw=visual_raw,
        audio_raw=audio_raw,
        min_dead_duration=min_dead,
        buffer_seconds=0.5,
        merge_gap=3.0,
    )
    active_dur = sum(s.duration for s in active_segs)
    dead_dur = duration - active_dur
    print(
        f"  Active content: {active_dur / 60:.1f} min across {len(active_segs)} segments "
        f"({dead_dur / 60:.1f} min dead air removed in {time.monotonic() - t0:.1f}s)"
    )

    parts = partition_active_stream(
        active_segs,
        transcript_segments=transcript_segments,
        cut_times=visual_raw.get("cut_times"),
        motion=visual_raw.get("motion"),
        loudness=audio_raw.get("loudness"),
        target_part_duration=target_dur,
        elastic_slack=slack,
        min_tail_duration=min_tail,
    )
    print(f"  Partitioned into {len(parts)} sequential part(s) (target 2-3 min each)")

    rendered: list[RenderedClip] = []
    clip_dir.mkdir(parents=True, exist_ok=True)

    for part in parts:
        cancel.check_active()
        part_path = clip_dir / part.label
        print(f"    Rendering {part.label} ({part.total_duration:.1f}s, {len(part.segments)} active cut(s))...")
        t_render = time.monotonic()

        render_video_part(video_path, part, part_path, config=config)
        print(f"    ✓ {part.label} rendered in {time.monotonic() - t_render:.1f}s")

        cand = ClipCandidate(
            start=part.start,
            end=part.end,
            score=90,
            hook=f"Part {part.part_number:02d}",
            reason=f"Video to Clips part {part.part_number:02d} ({part.total_duration:.0f}s)",
            source="video_to_clips",
        )
        clip = RenderedClip(source_video_id=video_id, candidate=cand, path=part_path)
        rendered.append(clip)

        if db is not None and video_id:
            try:
                from analysis.metadata import ClipMetadata
                from core.pipeline import _free_end, _register_clip

                end_s = _free_end(db, video_id, part.start, part.end)
                cand.end = end_s
                meta = ClipMetadata(
                    title=f"Part {part.part_number:02d}",
                    description=f"Part {part.part_number:02d} ({part.total_duration:.0f}s) of full recording.",
                    hashtags=[f"#part{part.part_number:02d}", "#series", "#fullvideo"],
                )
                _register_clip(
                    db,
                    video_id,
                    cand,
                    part_path,
                    meta,
                    json.dumps({"mode": "video_to_clips", "part": part.part_number}),
                    config,
                )
            except Exception as e:
                print(f"      (db registration skipped: {e})")

    return rendered
