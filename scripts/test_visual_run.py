"""Test runner to demonstrate visual frame analysis on a local video file."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.fusion import _build_events
from analysis.visual_captioning import describe_visual_events, resolve_vision_backend
from analysis.visual_features import extract_visual_features
import yaml

DEFAULT_VIDEO = r"C:\Projects\GTA 6\GTA 6 WTF and Funny Moments! #1.mp4"


def main():
    video_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    print(f"=== Testing Visual Frame Analysis on: {video_path.name} ===")

    # Load configuration
    config_file = Path("config/settings.yaml")
    config = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    vis_cfg = config.get("visual_analysis", {
        "enabled": True,
        "interval_seconds": 3.0,
        "max_frames": 20,
        "model": "",
    })

    print(f"Config: interval={vis_cfg.get('interval_seconds')}s, max_frames={vis_cfg.get('max_frames')}")

    # 1. Extract visual features (motion, scene_cuts)
    print("\n[1/3] Extracting fast visual signals (motion / cuts)...")
    t0 = time.monotonic()
    visual_raw = extract_visual_features(video_path)
    print(f"      Visual signals extracted in {time.monotonic() - t0:.1f}s")

    # 2. Run visual captioning with local vision model
    print("\n[2/3] Analyzing frames with Vision LLM (Ollama)...")
    llm = resolve_vision_backend(None, vis_cfg, config)
    print(f"      Selected Vision Model: {getattr(llm, 'name', 'None')}")

    captions = describe_visual_events(video_path, visual_raw, llm, vis_cfg, config)

    # 3. Show detected visual highlights
    print("\n[3/3] Standout Visual Highlights:")
    if not captions:
        print("      No standout visual events detected (all frames were judged NORMAL).")
    else:
        for t, desc in captions:
            print(f"      [{t:5.1f}s] {desc}")

    print("\n=== Test Run Finished Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
