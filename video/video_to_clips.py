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
import re
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
from video.encoding import concat, concat_entry, final_movflags, video_encoder_args


def parse_custom_chapters(text: str | None) -> list[tuple[float, str]]:
    """Parse custom timestamps/chapters into (seconds, label) tuples.

    Universally supports:
      - YouTube markdown links: [0:29](https://youtube.com/...) Mistake #1
      - Bracketed timestamps: [0:29] Mistake #1
      - Standard timestamps: 0:29 Mistake #1 or 00:01:45 Step 1
      - Multiline or continuous single-line strings.
    """
    if not text:
        return []
    pattern = re.compile(r"\[?(\d{1,2}(?::\d{2}){1,2})\]?(?:\([^\)]*\))?\s*[-:–—]?\s*")
    matches = list(pattern.finditer(text))
    chapters: list[tuple[float, str]] = []
    for i, m in enumerate(matches):
        ts_str = m.group(1)
        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_title = text[start_pos:end_pos].strip()
        # Clean URLs and markdown formatting from title
        clean_title = re.sub(r"https?://\S+", "", raw_title)
        clean_title = re.sub(r"\[.*?\]\(.*?\)", "", clean_title).strip(" -:–—\r\n\t")
        parts = ts_str.split(":")
        if len(parts) == 3:
            secs = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            secs = float(parts[0]) * 60 + float(parts[1])
        else:
            secs = float(parts[0])
        chapters.append((round(secs, 2), clean_title))
    return sorted(chapters, key=lambda x: x[0])


def format_chapter_titles(titles: Sequence[str]) -> str:
    """Format multiple chapter titles into a concise summary title."""
    c = [t.strip() for t in titles if t.strip()]
    if not c:
        return ""
    if len(c) == 1:
        return c[0]
    if len(c) == 2:
        return f"{c[0]} & {c[1]}"
    return f"{c[0]} - {c[-1]}"


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
    chapter_title: str | None = None

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
    chapter_timestamps: Sequence[float] | None = None,
) -> float:
    """Find the best natural split point within [window_start, window_end].

    Priority:
      0. Custom chapter timestamp provided by user
      1. Transcript sentence boundary (end of sentence with '.', '?', '!' or speech gap > 0.5s)
      2. Scene cut point
      3. Valley in combined motion & audio
      4. Fallback: window_start (or midpoint)
    """
    if window_end <= window_start:
        return window_start

    midpoint = (window_start + window_end) / 2.0

    # 0. User-provided chapter timestamp (highest priority)
    if chapter_timestamps:
        valid_chapters = [c for c in chapter_timestamps if window_start <= c <= window_end]
        if valid_chapters:
            best_ch = min(valid_chapters, key=lambda c: abs(c - midpoint))
            return round(best_ch, 2)

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
    custom_chapters: str | list[tuple[float, str]] | None = None,
) -> list[VideoPart]:
    """Partition active footage into sequential VideoParts of 120-180s.

    - Elastic extension: up to target_part_duration + elastic_slack (e.g. 195s)
      to snap to natural speech/scene/energy boundaries.
    - No mini-tail: if remaining active footage is < min_tail_duration (45s),
      merges it into the previous part.
    - Custom chapters: user timestamps take top priority as snap split boundaries.
    """
    if not active_segments:
        return []

    parsed_chapters: list[tuple[float, str]] = (
        parse_custom_chapters(custom_chapters) if isinstance(custom_chapters, str) else (custom_chapters or [])
    )
    chapter_ts = [c[0] for c in parsed_chapters]

    # Chapter-driven bin-packing mode (when chapters/timestamps provided)
    if parsed_chapters:
        max_part_dur = 180.0  # Hard 3-minute cap for YouTube Shorts
        ordered_ch = list(parsed_chapters)
        if ordered_ch[0][0] > 0.5:
            ordered_ch.insert(0, (0.0, "Intro"))

        video_end_t = active_segments[-1].end if active_segments else ordered_ch[-1][0] + 180.0
        ch_items: list[tuple[float, float, str, list[ActiveSegment], float]] = []

        for i in range(len(ordered_ch)):
            c_start = ordered_ch[i][0]
            c_end = ordered_ch[i + 1][0] if i + 1 < len(ordered_ch) else video_end_t
            c_title = ordered_ch[i][1] or f"Chapter {i + 1}"

            c_segs: list[ActiveSegment] = []
            for s in active_segments:
                if s.end > c_start and s.start < c_end:
                    c_segs.append(ActiveSegment(round(max(c_start, s.start), 3), round(min(c_end, s.end), 3)))
            c_segs = [s for s in c_segs if s.duration > 0.05]
            c_dur = sum(s.duration for s in c_segs)
            ch_items.append((c_start, c_end, c_title, c_segs, c_dur))

        parts: list[VideoPart] = []
        cur_titles: list[str] = []
        cur_segs: list[ActiveSegment] = []
        cur_dur = 0.0

        for c_start, c_end, c_title, c_segs, c_dur in ch_items:
            # Single huge chapter (> 180s): sub-partition so parts stay <= 180s
            if c_dur > max_part_dur:
                if cur_segs:
                    parts.append(
                        VideoPart(
                            part_number=len(parts) + 1,
                            segments=cur_segs,
                            start=cur_segs[0].start,
                            end=cur_segs[-1].end,
                            total_duration=round(cur_dur, 2),
                            chapter_title=format_chapter_titles(cur_titles),
                        )
                    )
                    cur_titles, cur_segs, cur_dur = [], [], 0.0

                sub_parts = partition_active_stream(
                    c_segs,
                    transcript_segments=transcript_segments,
                    cut_times=cut_times,
                    motion=motion,
                    loudness=loudness,
                    target_part_duration=180.0,
                    elastic_slack=0.0,
                    min_tail_duration=min_tail_duration,
                    custom_chapters=None,
                )
                for sp in sub_parts:
                    sub_title = f"{c_title} (Part {sp.part_number})" if len(sub_parts) > 1 else c_title
                    parts.append(
                        VideoPart(
                            part_number=len(parts) + 1,
                            segments=sp.segments,
                            start=sp.start,
                            end=sp.end,
                            total_duration=sp.total_duration,
                            chapter_title=sub_title,
                        )
                    )
                continue

            # If adding this chapter exceeds hard 180s limit, close current part!
            if cur_segs and (cur_dur + c_dur > max_part_dur):
                parts.append(
                    VideoPart(
                        part_number=len(parts) + 1,
                        segments=cur_segs,
                        start=cur_segs[0].start,
                        end=cur_segs[-1].end,
                        total_duration=round(cur_dur, 2),
                        chapter_title=format_chapter_titles(cur_titles),
                    )
                )
                cur_titles, cur_segs, cur_dur = [], [], 0.0

            if c_segs:
                cur_titles.append(c_title)
                cur_segs.extend(c_segs)
                cur_dur += c_dur

        if cur_segs:
            parts.append(
                VideoPart(
                    part_number=len(parts) + 1,
                    segments=cur_segs,
                    start=cur_segs[0].start,
                    end=cur_segs[-1].end,
                    total_duration=round(cur_dur, 2),
                    chapter_title=format_chapter_titles(cur_titles),
                )
            )

        return parts

    total_active = sum(s.duration for s in active_segments)
    if total_active <= (target_part_duration + elastic_slack + min_tail_duration):
        first_title = parsed_chapters[0][1] if parsed_chapters else None
        return [
            VideoPart(
                part_number=1,
                segments=active_segments,
                start=active_segments[0].start,
                end=active_segments[-1].end,
                total_duration=round(total_active, 2),
                chapter_title=first_title,
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
                chapter_timestamps=chapter_ts,
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
            matching_title = None
            if parsed_chapters:
                p_start = part_segments[0].start
                p_end = part_segments[-1].end
                # Find chapter that lands in this part
                for ch_t, ch_title in parsed_chapters:
                    if p_start - 2.0 <= ch_t <= p_end:
                        matching_title = ch_title
                        break

            parts.append(
                VideoPart(
                    part_number=len(parts) + 1,
                    segments=part_segments,
                    start=part_segments[0].start,
                    end=part_segments[-1].end,
                    total_duration=round(accum_dur, 2),
                    chapter_title=matching_title,
                )
            )

    return parts


def build_part_captions(
    segments: Sequence[Segment],
    part: VideoPart,
    output_path: Path,
    style: dict | None = None,
    canvas: tuple[int, int] = (1080, 1920),
    language: str = "en",
) -> Path | None:
    """Build an ASS subtitle file mapped to the concatenated timeline of VideoPart."""
    if not segments or not part.segments:
        return None
    try:
        from video.captions import build_captions

        remapped_segments: list[Segment] = []
        rendered_offset = 0.0

        for a_seg in part.segments:
            seg_dur = a_seg.duration
            for s in segments:
                if s.end > a_seg.start and s.start < a_seg.end:
                    clamped_start = max(a_seg.start, s.start)
                    clamped_end = min(a_seg.end, s.end)
                    rel_start = rendered_offset + (clamped_start - a_seg.start)
                    rel_end = rendered_offset + (clamped_end - a_seg.start)

                    rem_words = []
                    if getattr(s, "words", None):
                        for w in s.words:
                            w_s = w.get("start", clamped_start)
                            w_e = w.get("end", clamped_end)
                            if w_e > a_seg.start and w_s < a_seg.end:
                                w_clamped_s = max(a_seg.start, w_s)
                                w_clamped_e = min(a_seg.end, w_e)
                                rem_words.append({
                                    **w,
                                    "start": rendered_offset + (w_clamped_s - a_seg.start),
                                    "end": rendered_offset + (w_clamped_e - a_seg.start),
                                })
                    remapped_segments.append(
                        Segment(
                            start=round(rel_start, 3),
                            end=round(rel_end, 3),
                            text=s.text,
                            words=rem_words if rem_words else None,
                        )
                    )
            rendered_offset += seg_dur

        if not remapped_segments:
            return None

        dummy_cand = ClipCandidate(start=0.0, end=part.total_duration, score=100)
        return build_captions(
            remapped_segments,
            dummy_cand,
            output_path,
            style=style,
            canvas=canvas,
            language=language,
        )
    except Exception as e:
        print(f"      (build_part_captions failed: {e})")
        return None


def render_video_part(
    video_path: Path,
    part: VideoPart,
    output_path: Path,
    config: dict | None = None,
    crop_filter: str | None = None,
    ass_path: Path | None = None,
) -> Path:
    """Render a VideoPart into a continuous MP4 file with 40ms micro audio-crossfades.

    If not longform (default): formats as 9:16 vertical (1080x1920) with blurred background wash.
    If longform: maintains original 16:9 aspect ratio.
    If ass_path provided: burns subtitles into output stream.
    Compatible with FFmpeg 5, 6, 7, 8 and 9.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build filter complex for N segments with 40ms audio crossfade / fade
    filter_parts: list[str] = []
    n_segs = len(part.segments)

    for i, seg in enumerate(part.segments):
        dur = seg.duration
        v_label = f"v{i}"
        a_label = f"a{i}"

        # Video trim
        if crop_filter:
            filter_parts.append(
                f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS,{crop_filter}[{v_label}];"
            )
        else:
            filter_parts.append(
                f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},setpts=PTS-STARTPTS[{v_label}];"
            )

        # Audio trim with 40ms (0.04s) micro-fade at head and tail
        fade_d = min(0.04, dur / 3.0)
        st_out = max(0.0, dur - fade_d)
        filter_parts.append(
            f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:ss=0:d={fade_d:.3f},afade=t=out:st={st_out:.3f}:d={fade_d:.3f}[{a_label}];"
        )

    # Concat filter: FFmpeg concat filter requires interleaved stream pads per segment: [v0][a0][v1][a1]...
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n_segs))
    filter_parts.append(f"{concat_inputs}concat=n={n_segs}:v=1:a=1[v_concat][a_out];")

    # Layout: 9:16 vertical with blurred wash top & bottom vs 16:9 longform
    is_longform = bool(config and config.get("clips", {}).get("longform"))
    if is_longform:
        last_v = "[v_concat]"
    else:
        bg = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,scale=135:240,gblur=sigma=12,scale=1080:1920:flags=bilinear"
        )
        fg = "scale=1080:-2:flags=lanczos,crop=w=1080:h=min(ih\\,1920)"
        filter_parts.append(
            f"[v_concat]split=2[bg_in][fg_in];"
            f"[bg_in]{bg}[bg];"
            f"[fg_in]{fg}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v_vertical];"
        )
        last_v = "[v_vertical]"

    if ass_path and Path(ass_path).exists():
        clean_ass = str(Path(ass_path).name)
        filter_parts.append(f"{last_v}subtitles=filename='{clean_ass}'[v_subbed];")
        last_v = "[v_subbed]"

    filter_str = "".join(filter_parts).rstrip(";")

    enc_args = video_encoder_args(config)
    resolved_video_path = Path(video_path).resolve()
    cmd = [
        ffmpeg(),
        "-y",
        "-v",
        "error",
        "-i",
        str(resolved_video_path),
        "-filter_complex",
        filter_str,
        "-map",
        last_v,
        "-map",
        "[a_out]",
        *enc_args,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        *final_movflags(),
        str(output_path.resolve()),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_path.parent.resolve()))
    if r.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg render failed for {output_path.name}: {r.stderr.strip()[-300:]}")

    return output_path


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
