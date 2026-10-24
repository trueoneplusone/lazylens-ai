"""Run local LazyLens inference, optionally in strict offline mode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lazylens import MobileNetEmbedder, analyze, available_frames, save_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locally screen image sequences and keep only informative frames"
    )
    parser.add_argument("input", type=Path, help="Folder containing image frames")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--semantic-threshold", type=float, default=0.05)
    parser.add_argument("--texture-threshold", type=float, default=0.04)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--min-sharpness", type=float, default=0.0)
    parser.add_argument("--offline", action="store_true",
                        help="Refuse to download weights if uncached")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()

    files = available_frames(args.input)
    if not files:
        parser.error(f"No supported image files found in {args.input}")

    embedder = MobileNetEmbedder(device=args.device, offline=args.offline)
    records = analyze(
        files, embedder, semantic_threshold=args.semantic_threshold,
        texture_threshold=args.texture_threshold, max_gap=args.max_gap,
        min_sharpness=args.min_sharpness,
    )
    summary = save_report(args.input, args.out, records, embedder.model_name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOpen self-contained visual report: {args.out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
