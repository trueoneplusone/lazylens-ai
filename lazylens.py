"""LazyLens: private, local keyframe selection using open MobileNetV3 weights."""
from __future__ import annotations

import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class Embedder(Protocol):
    model_name: str

    def encode(self, image: Image.Image) -> np.ndarray:
        ...


@dataclass(frozen=True)
class Decision:
    index: int
    filename: str
    keep: bool
    reason: str
    semantic_change: float
    texture_change: float
    sharpness: float
    elapsed_ms: float


class MobileNetEmbedder:
    """Open-weight torchvision feature extractor. No images leave this computer."""

    model_name = "torchvision/mobilenet_v3_small/imagenet1k_v1"

    def __init__(self, *, device: str = "auto", offline: bool = False):
        import torch
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if offline:
            filename = Path(urlparse(weights.url).path).name
            cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not cached.exists():
                raise FileNotFoundError(
                    f"Offline weights not cached: {cached}. "
                    "Run once without --offline to cache official open weights."
                )
        self.torch = torch
        self.device = torch.device(device)
        self.preprocess = weights.transforms()
        self.model = mobilenet_v3_small(weights=weights).to(self.device).eval()
        self.model.classifier = torch.nn.Identity()

    def encode(self, image: Image.Image) -> np.ndarray:
        batch = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            v = self.model(batch).reshape(-1).float().cpu().numpy()
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-12 else v


def available_frames(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Input folder missing: {folder}")
    return sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def texture_and_sharpness(image: Image.Image) -> tuple[np.ndarray, float]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    thumb = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    return thumb, sharpness


def cosine_change(current: np.ndarray, last: np.ndarray) -> float:
    if current.shape != last.shape:
        raise ValueError("Embedding dimensionality changed")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(last)):
        raise ValueError("Embedding contains non-finite numbers")
    norm = float(np.linalg.norm(current) * np.linalg.norm(last))
    if norm <= 1e-12:
        return 0.0
    similarity = float(np.dot(current, last) / norm)
    return max(0.0, min(2.0, 1.0 - similarity))


def texture_change(current: np.ndarray, last: np.ndarray) -> float:
    return float(np.abs(current.astype(np.float32) -
                        last.astype(np.float32)).mean() / 255.0)


def analyze(
    paths: list[Path],
    embedder: Embedder,
    *,
    semantic_threshold: float = 0.05,
    texture_threshold: float = 0.04,
    max_gap: int = 8,
    min_sharpness: float = 0.0,
) -> list[Decision]:
    if max_gap < 1:
        raise ValueError("max_gap must be >= 1")
    if not (0 <= semantic_threshold <= 2):
        raise ValueError("semantic_threshold must be between 0 and 2")
    if not (0 <= texture_threshold <= 1):
        raise ValueError("texture_threshold must be between 0 and 1")

    records: list[Decision] = []
    last_vector: np.ndarray | None = None
    last_texture: np.ndarray | None = None
    last_index = 0

    for i, path in enumerate(paths):
        t0 = time.perf_counter()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        current_vector = np.asarray(embedder.encode(image), dtype=np.float32)
        current_texture, sharpness = texture_and_sharpness(image)

        if last_vector is None:
            semantic, texture = 0.0, 0.0
            keep, reason = True, "first_frame"
        else:
            semantic = cosine_change(current_vector, last_vector)
            texture = texture_change(current_texture, last_texture)
            if min_sharpness > 0 and sharpness < min_sharpness:
                keep, reason = False, "blurry"
            elif i - last_index >= max_gap:
                keep, reason = True, "max_gap"
            elif semantic >= semantic_threshold:
                keep, reason = True, "semantic_change"
            elif texture >= texture_threshold:
                keep, reason = True, "texture_change"
            else:
                keep, reason = False, "redundant"

        if keep:
            last_vector = current_vector
            last_texture = current_texture
            last_index = i

        records.append(Decision(
            index=i, filename=path.name, keep=keep, reason=reason,
            semantic_change=round(semantic, 6),
            texture_change=round(texture, 6),
            sharpness=round(sharpness, 3),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        ))
    return records


def save_report(input_dir: Path, output_dir: Path,
                decisions: list[Decision], model_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [r for r in decisions if r.keep]
    total = len(decisions)
    saved = total - len(selected)
    summary = {
        "model": model_name,
        "frames_total": total,
        "frames_selected": len(selected),
        "frames_skipped": saved,
        "downstream_jobs_avoided_pct": round(100.0 * saved / total, 2) if total else 0.0,
        "embedding_and_selection_ms": round(sum(d.elapsed_ms for d in decisions), 1),
        "disclaimer": "Savings estimate excludes model inference and other selection overhead.",
        "selected_files": [d.filename for d in selected],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Decision.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(d) for d in decisions)

    import base64
    from io import BytesIO
    cards = []
    for d in decisions:
        with Image.open(input_dir / d.filename) as source:
            image = source.copy()
        image.thumbnail((176, 116))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=76)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        safe_filename = html.escape(d.filename)
        cards.append(
            f'<article class="{"keep" if d.keep else "skip"}">'
            f'<img alt="{safe_filename}" src="data:image/jpeg;base64,{encoded}">'
            f'<div><b>{safe_filename}</b><br>'
            f'<small>{html.escape(d.reason)} | sem {d.semantic_change:.3f} | '
            f'pix {d.texture_change:.3f}</small></div></article>'
        )
    report = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>LazyLens | offline AI keyframe report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:auto;max-width:1080px;padding:32px;color:#eaeaf0;background:#11131a;
font:16px/1.5 system-ui,sans-serif}h1{font-size:44px;margin:0}
p{color:#b7becd}.metrics{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}
.metric{background:#202633;border:1px solid #30394a;border-radius:16px;
padding:18px;min-width:160px}.metric strong{font-size:30px;color:#b7f7b7}
.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px}
article{background:#202633;border:2px solid #30394a;border-radius:12px;overflow:hidden}
article.keep{border-color:#76edac}article.skip{opacity:.46}
article img{width:100%;height:116px;object-fit:cover}
article div{padding:9px;font-size:12px;overflow-wrap:anywhere}
small{color:#b7becd}footer{margin-top:36px;color:#99a3b5}
</style>
<h1>LazyLens 🦥</h1><p>Let the model watch every frame. Your expensive
downstream pipeline only works on the important ones.</p>
<div class="metrics"><div class="metric">Frames<br><strong>TOTAL</strong></div>
<div class="metric">Selected<br><strong>SELECTED</strong></div>
<div class="metric">Avoided downstream jobs<br><strong>SAVINGS%</strong></div>
</div><div class="frames">CARDS</div>
<footer>Private local inference · MobileNetV3 open weights · every skipped frame
remains on disk. Savings exclude selector inference.</footer></html>"""
    report = (
        report.replace("TOTAL", str(total))
        .replace("SELECTED", str(len(selected)))
        .replace("SAVINGS%", f"{summary['downstream_jobs_avoided_pct']}%")
        .replace("CARDS", "\n".join(cards))
    )
    (output_dir / "report.html").write_text(report, encoding="utf-8")
    return summary
