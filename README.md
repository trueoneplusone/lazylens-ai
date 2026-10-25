# LazyLens 🦥 — Your AI watches the boring frames so you don't have to

A local, open-weight **keyframe selector** for camera folders and image sequences. Instead of running a costly downstream computer-vision pipeline on every frame, LazyLens picks likely-important frames while preserving the original image files.

Created during **Hacktoberfest 2026 Week 3: Slow and Steady** (October 19–25).

**Live visual demo:** https://trueoneplusone.github.io/lazylens-ai/ (GitHub Pages; self-contained static report, no image upload).
**DEV Week 3 write-up:** https://dev.to/trueoneplusone/lazylens-i-let-an-open-weight-ai-skip-the-frames-i-didnt-need-1a81

## Why this exists

Picture a fixed camera monitoring a manufacturing surface, microscope view, or static room: most frames barely change. Running expensive segmentation, tracking, inspection, or 3D analysis on each identical frame is wasted work. LazyLens compares each frame to the *last kept frame* using pretrained open-weight visual embeddings and a lightweight spatial texture signal; changes or a maximum frame gap trigger processing.

**Nothing is uploaded**. Images are processed locally by PyTorch/torchvision and OpenCV. The demo is fully synthetic, so no personal data is included.

## Measured result — including model overhead

Deterministic 32-frame synthetic sequence (two staged visual events), MobileNetV3-small ImageNet pretrained on CUDA device 0 (NVIDIA GeForce RTX 5090; four RTX 5090 GPUs installed), OpenCV Farneback downstream benchmark on CPU on the same Windows workstation, median of 3 runs:

| Metric | Result |
|---|---:|
| Input frames | 32 |
| Important/periodic frames selected | 5 |
| Expensive downstream jobs avoided | 27 / 32 = **84.38%** |
| Selection inference + decision overhead | **770.4 ms** |
| Baseline: one optical-flow run per frame | 660.4 ms |
| Total with selection, one run/frame | 872.8 ms (**32.16% slower**) |
| Baseline: three independent flow settings/frame | 1,798.3 ms |
| Total with selection, three runs/frame | 1,049.4 ms (**41.64% faster**) |

The 84.38% figure is *not* net speedup; it is the reduction in downstream frame count. For a lightweight downstream task, running the model costs more than it saves. For a sufficiently expensive pipeline, the filtering pays off. Selection also risks missing meaningful events, especially tiny/brief changes; the default max-gap forces periodic sampling but does not guarantee no missed defects.

These are **single-machine, synthetic-scene demonstration measurements**, not general performance claims. Re-run benchmarks for your own application. See `results/benchmark_passes_1.json` and `results/benchmark_passes_3.json`.

## Installation

Python 3.10+ recommended. Dependencies: `torch`, `torchvision`, `opencv-python`, `pillow`, `numpy`.

```bash
python -m pip install -r requirements.txt
python generate_demo.py
python run.py demo_frames --out results --max-gap 9 --semantic-threshold 0.08 --texture-threshold 0.025
```

The first run downloads the official ~10 MB torchvision MobileNetV3-small model weights and caches them. **After the first download, everything can run without internet:**

```bash
python run.py demo_frames --out results --offline --max-gap 9 --semantic-threshold 0.08 --texture-threshold 0.025
```

Open `results/report.html` in any browser: it is a self-contained local visual report with keep/skip labels; no cloud account or web server needed.

## Benchmark and validation

```bash
python -m unittest discover -s tests -v
python benchmark.py demo_frames results --repeat 3 --passes 1
python benchmark.py demo_frames results --repeat 3 --passes 3
```

In the benchmark, `--passes 3` means evaluating independent Farneback window sizes 9, 15 and 23 against each frame, to approximate a *heavier downstream analysis*. Each pass is actual OpenCV dense optical-flow computation; it is not simulated sleep. The selector time is included in the combined total.

## How it works

1. Sort images from the input directory and load each frame.
2. Extract a normalized MobileNetV3-small embedding using open ImageNet weights.
3. Compare cosine feature change against the last **kept** frame.
4. Compare the mean absolute change of a small grayscale thumbnail.
5. Keep a frame when semantics or appearance changes past a configurable threshold, or after `--max-gap` frames.
6. Export `decisions.csv`, `summary.json`, `report.html`; **never delete** originals.

### Controls

| Argument | Description |
|---|---|
| `--out PATH` | Output location |
| `--device auto\|cpu\|cuda` | Inference device |
| `--offline` | Reject uncached weights instead of downloading |
| `--semantic-threshold FLOAT` | Cosine embedding difference threshold |
| `--texture-threshold FLOAT` | Pixel-thumbnail change threshold |
| `--max-gap INT` | Upper bound on consecutive unselected frames |
| `--min-sharpness FLOAT` | Optional blur gate |

The standard image extensions are PNG/JPG/BMP/WebP/TIFF. Filenames are sorted lexicographically; name files with zero-padded numeric indices for chronological sequences.

## Privacy & honest limitations

- AI is **open-weight** and inference runs locally, including offline after model caching.
- No images, secrets, API keys or wallet information are uploaded by the tool.
- The demo represents simple staged changes in a static scene, **not** a certified safety- or quality-inspection system.
- A short-lived event could be missed and frame counts alone don't prove detection reliability.
- OpenCV benchmark is a proxy for expensive downstream analysis. General speedups depend on workload and hardware.
- This project was developed with AI assistance, locally tested, and deliberately reports unfavorable results as well as favorable ones.

## Open innovation

Being able to run open weights locally matters for industrial and research images that cannot be uploaded to a service. Users can replace MobileNet with their own model and audit the frame-selection logic. All generated artifacts are inspectable; the decision for every frame is recorded.

## Credits

- Model and preprocessing: PyTorch / torchvision MobileNetV3 Small ImageNet1K weights (official distribution).
- Vision primitives: OpenCV; image I/O: Pillow.
- Generated textures and defect events are synthetic. No third-party images embedded.

Code license: MIT (see LICENSE). Third-party dependencies/weights retain their own licenses.
