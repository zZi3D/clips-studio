"""Visual frame captioning for gameplay/action moments using a local Vision model.

Samples frames at fixed intervals and on visual activity peaks / scene cuts,
asking a local multimodal vision model (e.g. Gemma 4 / Gemma 3 via Ollama)
to describe standout action, crashes, glitches, or stunts in max 12 words.
Moments judged 'NORMAL' are dropped; standout moments are returned as
[(timestamp, "VISUAL: <description>")] to inform clip scoring.
"""

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from analysis.game_vision import _jpeg, can_see
from core import cancel, modes
from llm.base import LLMBackend
from video.capture import video_capture

PROMPT_ACTION = (
    "Describe in max 12 words if something funny, chaotic, a crash, explosion, "
    "glitch, ragdoll, or impressive stunt happens in this gameplay frame. "
    "Reply strictly 'NORMAL' if nothing special happens."
)

PROMPT_RECAP = (
    "Describe in max 14 words if this frame shows a tutorial step, map location, "
    "secret item, easter egg, objective, puzzle, crash, or key gameplay event. "
    "Reply strictly 'NORMAL' if it is only idle walking, driving, loading, or nothing notable happens."
)

PROMPT = PROMPT_ACTION


def _find_installed_vision_model(host: str) -> str | None:
    """Find the best available local model that supports vision from Ollama."""
    try:
        import requests

        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            for m in models:
                if "gemma4" in m.get("name", ""):
                    return m.get("name")
            for m in models:
                caps = m.get("capabilities", []) or (m.get("details", {}) or {}).get("capabilities", [])
                if "vision" in caps:
                    return m.get("name")
    except Exception:
        pass
    return None


def resolve_vision_backend(
    main_llm: LLMBackend,
    vis_cfg: dict,
    full_config: dict | None = None,
) -> LLMBackend | None:
    """Resolve which backend to use for vision tasks.

    Uses vis_cfg['model'] if specified. Otherwise uses main_llm if it sees
    images, or auto-detects an installed vision model (e.g. qwen3.5:9b / gemma4).
    """
    model_override = str(vis_cfg.get("model") or "").strip()
    ollama_host = (full_config or {}).get("llm", {}).get("ollama_host", "http://localhost:11434")

    if model_override:
        model_name = model_override
        if model_name.startswith("ollama/"):
            model_name = model_name.split("/", 1)[1]
        from llm.ollama_backend import OllamaBackend

        return OllamaBackend(model=model_name, host=ollama_host)

    try:
        if main_llm and main_llm.sees_images():
            return main_llm
    except Exception:
        pass

    from llm.ollama_backend import OllamaBackend

    detected = _find_installed_vision_model(ollama_host)
    return OllamaBackend(model=detected or "gemma4:e4b", host=ollama_host)


def select_frame_times(
    duration: float,
    visual_raw: dict | None = None,
    interval_seconds: float = 3.0,
    max_frames: int = 120,
) -> list[float]:
    """Select candidate timestamps across the video based on interval and visual peaks."""
    raw = visual_raw or {}
    duration = max(0.0, float(duration))
    if duration <= 0:
        motion = raw.get("motion")
        if motion is not None and len(motion) > 0:
            duration = float(len(motion))

    if duration <= 0:
        return []

    times: list[float] = []

    # 1. Periodic intervals
    if interval_seconds > 0:
        curr = 0.0
        while curr < duration:
            times.append(round(curr, 2))
            curr += interval_seconds

    # 2. Visual peaks & scene cuts
    scene_cuts = raw.get("scene_cut", np.zeros(0))
    for sec in range(len(scene_cuts)):
        if scene_cuts[sec] >= 1:
            times.append(float(sec))

    motion = raw.get("motion", np.zeros(0))
    if len(motion) > 0:
        thresh = float(np.percentile(motion, 90))
        for sec in range(len(motion)):
            if motion[sec] >= thresh and motion[sec] > 0:
                times.append(float(sec))

    # 3. Deduplicate (keep times at least 1.0s apart) and sort
    times.sort()
    deduped: list[float] = []
    for t in times:
        if not deduped or (t - deduped[-1] >= 1.0):
            deduped.append(t)

    # 4. Cap at max_frames while preserving even spread
    if len(deduped) > max_frames and max_frames > 0:
        step = len(deduped) / max_frames
        deduped = [deduped[int(i * step)] for i in range(max_frames)]

    return deduped


def _clean_caption(raw: str) -> str | None:
    """Normalize model response. Returns cleaned caption text or None if NORMAL."""
    if not raw:
        return None

    # Immediate rejection if 'NORMAL' is mentioned anywhere
    if "normal" in raw.lower():
        return None

    text = raw.strip()
    # Strip markdown code fences if wrapped in ```json or ```
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    # If returned as a JSON object, extract string value
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, str) and val.strip():
                        text = val
                        break
        except Exception:
            pass

    text = " ".join(text.split()).strip(' "\'')
    if not text or "normal" in text.lower():
        return None

    # Truncate if model exceeded length guidelines
    words = text.split()
    if len(words) > 16:
        text = " ".join(words[:16])

    return text


def describe_visual_events(
    video_path: Path | str,
    visual_raw: dict,
    llm: LLMBackend,
    config: dict | None = None,
    full_config: dict | None = None,
) -> list[tuple[float, str]]:
    """Inspect video frames with a local vision model and generate visual event tags.

    Returns:
        list of (timestamp_seconds, "VISUAL: <description>")
    """
    vis_cfg = (config or {}).get("visual_analysis", config or {})
    if not vis_cfg.get("enabled", True):
        return []

    interval = float(vis_cfg.get("interval_seconds", 3.0))
    max_frames = int(vis_cfg.get("max_frames", 120))

    recap_active = modes.is_recap(full_config)
    video_to_clips_active = modes.is_video_to_clips(full_config)

    if video_to_clips_active:
        v2c_cfg = (full_config or {}).get("video_to_clips_mode", {})
        interval = float(v2c_cfg.get("frame_interval_seconds", 8.0))
        max_frames = int(v2c_cfg.get("max_frames", 1200))
    elif recap_active:
        rcp_cfg = (full_config or {}).get("recap_mode", {})
        interval = float(rcp_cfg.get("frame_interval_seconds", interval))
        max_frames = int(rcp_cfg.get("max_frames", max_frames))

    prompt = PROMPT_RECAP if recap_active else PROMPT_ACTION

    vision_backend = resolve_vision_backend(llm, vis_cfg, full_config)
    if vision_backend is None:
        return []

    # Check if vision backend can see images
    try:
        if not vision_backend.sees_images():
            print(f"  Visual frame analysis: {vision_backend.name} does not advertise vision, skipping")
            return []
        if not can_see(vision_backend):
            print(f"  Visual frame analysis: {vision_backend.name} failed visual capability check, skipping")
            return []
    except Exception as e:
        print(f"  Visual frame analysis: vision check failed ({e}), skipping")
        return []

    # Compute video duration and frame sample times
    duration = 0.0
    with video_capture(video_path, required=False) as cap:
        if cap is None:
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps

    times = select_frame_times(duration, visual_raw, interval_seconds=interval, max_frames=max_frames)
    if not times:
        return []

    print(f"  Analyzing {len(times)} visual frame(s) with {vision_backend.name}...")
    t0 = time.monotonic()
    results: list[tuple[float, str]] = []

    with video_capture(video_path, required=False) as cap:
        if cap is None:
            return []

        for t in times:
            cancel.check_active()
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            img_bytes = _jpeg(frame)
            if not img_bytes:
                continue

            try:
                raw_answer = vision_backend.look(prompt, [img_bytes], json_mode=False)
            except Exception as e:
                # Log once or continue gracefully on server/timeout issues
                print(f"    (frame at {t:.1f}s look failed: {e})")
                continue

            caption = _clean_caption(raw_answer)
            if caption:
                results.append((round(t, 1), f"VISUAL: {caption}"))

    elapsed = time.monotonic() - t0
    if results:
        print(f"  Captured {len(results)} standout visual event(s) in {elapsed:.1f}s")
    else:
        print(f"  Visual frame analysis completed in {elapsed:.1f}s (no standout visual events detected)")

    return sorted(results, key=lambda x: x[0])
