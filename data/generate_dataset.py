"""
VIGIL-CV :: Synthetic dataset generator

Generates a small, fully-offline "computer vision" dataset of simple shapes
(circle / square / triangle) so the whole VIGIL-CV demo never depends on
downloading real photos. Each image is tagged with a contributor_id and
batch_id so F1 can demonstrate contributor/batch-level risk aggregation.

This is intentionally simple (32x32 RGB) so a tiny model can be hand-built
and trained instantly on CPU with no heavyweight ML framework.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle"]
IMG_SIZE = 32

# Each class has a "true" color family so a simple color-based classifier
# has real signal to learn from.
CLASS_COLORS = {
    "circle": [(200, 40, 40), (220, 60, 60), (180, 30, 30)],      # reds
    "square": [(40, 120, 200), (60, 140, 220), (30, 100, 180)],   # blues
    "triangle": [(40, 180, 80), (60, 200, 100), (30, 160, 70)],   # greens
}

CONTRIBUTORS = ["contributor_A", "contributor_B", "contributor_C"]


def _draw_shape(draw, shape, color, jitter=9):
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = int(IMG_SIZE // 3 * random.uniform(0.75, 1.15))
    dx, dy = random.randint(-jitter, jitter), random.randint(-jitter, jitter)
    cx, cy = cx + dx, cy + dy
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        angle = random.uniform(-0.35, 0.35)
        pts = [(-r, -r), (r, -r), (r, r), (-r, r)]
        pts = [
            (cx + p[0] * np.cos(angle) - p[1] * np.sin(angle),
             cy + p[0] * np.sin(angle) + p[1] * np.cos(angle))
            for p in pts
        ]
        draw.polygon(pts, fill=color)
    elif shape == "triangle":
        angle = random.uniform(-0.35, 0.35)
        pts = [(0, -r), (-r, r), (r, r)]
        pts = [
            (cx + p[0] * np.cos(angle) - p[1] * np.sin(angle),
             cy + p[0] * np.sin(angle) + p[1] * np.cos(angle))
            for p in pts
        ]
        draw.polygon(pts, fill=color)


def make_image(label, seed_color=None, bg_noise=True):
    bg = random.randint(235, 255)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    if bg_noise:
        for _ in range(random.randint(15, 40)):
            x, y = random.randint(0, IMG_SIZE - 1), random.randint(0, IMG_SIZE - 1)
            g = random.randint(190, 250)
            draw.point((x, y), fill=(g, g, g))
    color = seed_color or random.choice(CLASS_COLORS[label])
    jitter_color = tuple(max(0, min(255, c + random.randint(-12, 12))) for c in color)
    _draw_shape(draw, label, jitter_color)
    return img


def generate(out_dir: str, n_per_class: int = 40, seed: int = 42):
    random.seed(seed)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    sample_idx = 0
    for label in CLASSES:
        for i in range(n_per_class):
            contributor = CONTRIBUTORS[sample_idx % len(CONTRIBUTORS)]
            batch_id = f"batch_{(sample_idx // 10) % 5}"
            img = make_image(label)
            fname = f"{label}_{i:04d}.png"
            img.save(img_dir / fname)
            records.append({
                "sample_id": f"s{sample_idx:05d}",
                "filename": fname,
                "label": label,
                "contributor_id": contributor,
                "batch_id": batch_id,
            })
            sample_idx += 1

    with open(out_dir / "labels.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {len(records)} images -> {img_dir}")
    return records


if __name__ == "__main__":
    generate(os.path.join(os.path.dirname(__file__), "clean_dataset"), n_per_class=40)
