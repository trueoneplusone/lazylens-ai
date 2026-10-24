"""Build an accessible poster from deterministic locally generated demo frames."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "overview.png"
FRAMES = [0, 9, 11, 23]
TAGS = ["INITIAL", "PERIODIC", "DEFECT A", "DEFECT B"]
BASE = (15, 19, 29)
FG = (232, 242, 248)
MUTED = (158, 177, 193)


def font(size: int):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1200, 630), BASE)
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 45), "LazyLens", font=font(60), fill=FG)
    draw.text((62, 119), "Let local open-weight AI skip the boring frames.",
              font=font(25), fill=MUTED)
    draw.text((62, 174), "32 input frames        5 selected        84.38% fewer downstream jobs",
              font=font(23), fill=(139, 225, 170))
    for i, (frame, tag) in enumerate(zip(FRAMES, TAGS)):
        x = 58 + i * 290
        with Image.open(ROOT / "demo_frames" / f"frame_{frame:03d}.png") as source:
            thumb = source.convert("RGB").resize((264, 164))
        canvas.paste(thumb, (x, 247))
        draw.rounded_rectangle((x - 2, 245, x + 265, 413),
                               radius=8, outline=(93, 194, 147), width=3)
        draw.text((x, 425), f"#{frame:03d}  {tag}", font=font(20), fill=FG)
    draw.text((62, 513), "Actual total latency: -32% on cheap analysis  |  +42% on heavier 3-pass analysis",
              font=font(21), fill=(249, 209, 148))
    draw.text((62, 552), "Synthetic demo  -  private local inference  -  honest benchmark",
              font=font(19), fill=MUTED)
    canvas.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
