"""Generate licensed, deterministic demo data; no real people or uploaded images."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SEED = 4242026


def main() -> None:
    root = Path(__file__).resolve().parent
    dest = root / "demo_frames"
    dest.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    height, width = 260, 420

    # Repeat deliberately: many sources record identical images between events.
    image = (105 + rng.normal(0, 32, (height, width))).clip(0, 255).astype("uint8")
    base = Image.fromarray(image, mode="L").convert("RGB")
    painter = ImageDraw.Draw(base)
    painter.rectangle((20, 20, 180, 74), fill=(21, 35, 55))
    painter.text((30, 38), "Quiet / No change", fill=(245, 250, 255))

    for index in range(32):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        if 11 <= index < 23:
            # A sudden localized defect. It should be retained by a frame selector.
            draw.ellipse((210, 75, 295, 165), fill=(240, 94, 69))
            draw.text((220, 188), "DEFECT A", fill=(15, 20, 24))
        elif index >= 23:
            draw.ellipse((245, 58, 337, 163), fill=(240, 94, 69))
            draw.line((211, 190, 378, 220), width=9, fill=(78, 181, 243))
            draw.text((220, 233), "DEFECT B", fill=(15, 20, 24))
        # No blurs/no injected stochastic variation: exact duplicates are meaningful.
        frame.save(dest / f"frame_{index:03d}.png")

    print(f"Generated 32 synthetic frames in {dest}")


if __name__ == "__main__":
    main()
