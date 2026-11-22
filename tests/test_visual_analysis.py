import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from analysis.visual_captioning import select_frame_times, _clean_caption, resolve_vision_backend
from analysis.fusion import _build_events
from llm.base import LLMBackend

def test_select_frame_times():
    visual_raw = {
        'motion': np.array([1.0, 2.0, 50.0, 1.0, 2.0, 1.0]),
        'scene_cut': np.array([0, 0, 0, 2, 0, 0])
    }
    times = select_frame_times(6.0, visual_raw, interval_seconds=2.0, max_frames=10)
    print("Times:", times)
    assert 0.0 in times
    assert 2.0 in times
    assert 4.0 in times
    assert any(t in (2.0, 3.0) for t in times)
    assert len(times) <= 10

def test_clean_caption():
    assert _clean_caption("NORMAL") is None
    assert _clean_caption("normal") is None
    assert _clean_caption("NORMAL.") is None
    assert _clean_caption("It looks normal to me") is None
    assert _clean_caption("A car flips over after a massive explosion") == "A car flips over after a massive explosion"
    assert _clean_caption('{"description": "A player does a backflip"}') == "A player does a backflip"
    assert _clean_caption('```json\n{"text": "Car crashes into wall"}\n```') == "Car crashes into wall"

def test_resolve_vision_backend():
    class MockTextLLM(LLMBackend):
        def generate(self, p, **k): return ""
        @property
        def name(self): return "mock/text"

    class MockVisionLLM(LLMBackend):
        def generate(self, p, **k): return ""
        def sees_images(self): return True
        @property
        def name(self): return "mock/vision"

    text_llm = MockTextLLM()
    vision_llm = MockVisionLLM()

    b1 = resolve_vision_backend(text_llm, {"model": ""})
    assert getattr(b1, "model", None) in ("gemma4:e4b", "qwen3.5:9b", "qwen3.5-mcp:latest", "qwen3-vl:8b")
    b2 = resolve_vision_backend(vision_llm, {"model": ""})
    assert b2 == vision_llm
    b3 = resolve_vision_backend(text_llm, {"model": "minicpm-v"})
    assert getattr(b3, "model", None) == "minicpm-v"

def test_build_events():
    audio = np.array([0.1, 0.99, 0.1])
    visual = np.array([0.1, 0.1, 0.99])
    captions = [(1.5, "VISUAL: funny ragdoll physics")]
    events = _build_events(audio, visual, {}, visual_captions=captions)
    print("Events:", events)
    assert any("VISUAL:" in desc for t, desc in events)
    assert events == sorted(events, key=lambda x: x[0])

if __name__ == "__main__":
    test_select_frame_times()
    test_clean_caption()
    test_resolve_vision_backend()
    test_build_events()
    print("ALL TESTS PASSED SUCCESSFULLY!")
