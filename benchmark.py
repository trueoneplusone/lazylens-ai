"""Measure the real cost of local dense optical-flow downstream processing."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def flow_workload(folder: Path, names: list[str], repetitions: int = 3, passes: int = 1) -> float:
    if not names:
        return 0.0
    with Image.open(folder / names[0]) as first:
        start = cv2.cvtColor(np.asarray(first.convert("RGB")), cv2.COLOR_RGB2GRAY)
    elapsed = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        for name in names:
            with Image.open(folder / name) as f:
                gray = cv2.cvtColor(np.asarray(f.convert("RGB")), cv2.COLOR_RGB2GRAY)
            for window in (9, 15, 23, 31)[:passes]:
                result = cv2.calcOpticalFlowFarneback(
                    start, gray, None, pyr_scale=0.5, levels=3,
                    winsize=window, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                assert result.shape == (*gray.shape, 2)
        elapsed.append(1000 * (time.perf_counter() - t0))
    return round(statistics.median(elapsed), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("result", type=Path)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--passes", type=int, choices=(1, 2, 3, 4), default=1,
                   help="Run independent optical-flow parameter settings per frame")
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be positive")
    from lazylens import available_frames
    summary = json.loads((args.result / "summary.json").read_text(encoding="utf-8"))
    all_names = [f.name for f in available_frames(args.input)]
    selected_names = summary["selected_files"]
    if not all_names or not set(selected_names).issubset(set(all_names)):
        p.error("Report and input images do not match")
    # Warm up OpenCV to avoid mixing initialization with workload.
    flow_workload(args.input, all_names[:1], repetitions=1)
    baseline_ms = flow_workload(args.input, all_names, repetitions=args.repeat, passes=args.passes)
    selected_ms = flow_workload(args.input, selected_names, repetitions=args.repeat, passes=args.passes)
    selection_ms = float(summary["embedding_and_selection_ms"])
    combined_ms = round(selected_ms + selection_ms, 1)
    measured = {
        "workload": "OpenCV dense Farneback optical flow against first image",
        "independent_window_sizes": [9, 15, 23, 31][:args.passes],
        "median_of_runs": args.repeat,
        "baseline_all_frames_ms": baseline_ms,
        "downstream_selected_frames_ms": selected_ms,
        "ai_selection_ms": selection_ms,
        "combined_selected_ms": combined_ms,
        "net_saved_pct": round(100 * (baseline_ms - combined_ms) / baseline_ms, 2),
        "break_even_downstream_ms_per_skipped_frame": round(
            selection_ms / max(1, len(all_names) - len(selected_names)), 2),
        "notes": [
            "Measured on a synthetic example and this one PC, not claimed universal.",
            "Optical flow is a downstream proxy, not the AI selector's own accuracy.",
            "Overhead depends strongly on input size, GPU/CPU, and application.",
        ],
    }
    (args.result / f"benchmark_passes_{args.passes}.json").write_text(
        json.dumps(measured, indent=2), encoding="utf-8"
    )
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
