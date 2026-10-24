"""Deterministic, offline unit tests: no weights download or network."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazylens import (
    analyze,
    available_frames,
    cosine_change,
    save_report,
    texture_change,
)


class MeanColorEmbedder:
    model_name = "test/mean_color"

    def encode(self, image: Image.Image) -> np.ndarray:
        value = np.asarray(image, dtype="float32").mean(axis=(0, 1))
        return value / max(1e-9, np.linalg.norm(value))


class FixedEmbedder:
    model_name = "test/fixed"

    def encode(self, image: Image.Image) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype="float32")


def write_images(root: Path, colors: list[tuple[int, int, int]]) -> list[Path]:
    result = []
    for i, color in enumerate(colors):
        p = root / f"frame_{i:03d}.png"
        Image.new("RGB", (48, 48), color=color).save(p)
        result.append(p)
    return result


class LazyLensTests(unittest.TestCase):
    def test_first_and_duplicates(self):
        with tempfile.TemporaryDirectory() as dirname:
            p = write_images(Path(dirname), [(100, 50, 20)] * 5)
            result = analyze(p, MeanColorEmbedder(), max_gap=9)
            self.assertEqual([d.keep for d in result], [True, False, False, False, False])
            self.assertEqual(result[1].reason, "redundant")

    def test_max_gap_cannot_skip_forever(self):
        with tempfile.TemporaryDirectory() as dirname:
            p = write_images(Path(dirname), [(90, 90, 90)] * 7)
            result = analyze(p, FixedEmbedder(), max_gap=3)
            self.assertEqual([d.index for d in result if d.keep], [0, 3, 6])

    def test_semantic_change(self):
        with tempfile.TemporaryDirectory() as dirname:
            p = write_images(Path(dirname), [(250, 0, 0), (0, 0, 250)])
            result = analyze(p, MeanColorEmbedder(), semantic_threshold=0.1)
            self.assertEqual(result[1].reason, "semantic_change")

    def test_texture_change_with_fixed_embedder(self):
        with tempfile.TemporaryDirectory() as dirname:
            p = write_images(Path(dirname), [(10, 10, 10), (240, 240, 240)])
            result = analyze(p, FixedEmbedder(), texture_threshold=0.1)
            self.assertEqual(result[1].reason, "texture_change")

    def test_blur_gate(self):
        with tempfile.TemporaryDirectory() as dirname:
            p = write_images(Path(dirname), [(30, 30, 30)] * 4)
            result = analyze(p, FixedEmbedder(), max_gap=2, min_sharpness=100.0)
            self.assertEqual([d.keep for d in result], [True, False, False, False])
            self.assertEqual(result[-1].reason, "blurry")

    def test_bad_thresholds(self):
        with self.assertRaises(ValueError):
            analyze([], FixedEmbedder(), max_gap=0)
        with self.assertRaises(ValueError):
            analyze([], FixedEmbedder(), semantic_threshold=3.0)
        with self.assertRaises(ValueError):
            analyze([], FixedEmbedder(), texture_threshold=-0.1)

    def test_cosine_change_zero_and_identical(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        self.assertEqual(cosine_change(a, a), 0.0)
        self.assertEqual(cosine_change(a, b), 0.0)
        self.assertAlmostEqual(cosine_change(a, np.array([-1.0, 0.0])), 2.0)

    def test_texture_delta(self):
        a = np.zeros((96, 96), dtype=np.uint8)
        b = np.full((96, 96), 255, dtype=np.uint8)
        self.assertEqual(texture_change(a, b), 1.0)

    def test_report_and_escaping(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            p = write_images(root, [(50, 50, 50), (50, 50, 50)])
            records = analyze(p, FixedEmbedder())
            output = root / "out"
            data = save_report(root, output, records, "test/mock")
            self.assertEqual(data["frames_skipped"], 1)
            self.assertEqual(data["downstream_jobs_avoided_pct"], 50.0)
            with (output / "decisions.csv").open(encoding="utf-8", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            self.assertIn("LazyLens", (output / "report.html").read_text(encoding="utf-8"))
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["frames_selected"], 1)

    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            result = save_report(root, root / "out", [], "test/mock")
            self.assertEqual(result["frames_total"], 0)
            self.assertEqual(result["downstream_jobs_avoided_pct"], 0.0)

    def test_available_frames(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            write_images(root, [(1, 2, 3)])
            (root / "note.txt").write_text("ignore")
            self.assertEqual(len(available_frames(root)), 1)


if __name__ == "__main__":
    unittest.main()
