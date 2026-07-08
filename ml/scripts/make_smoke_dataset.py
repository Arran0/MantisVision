"""Generate a tiny synthetic dataset to smoke-test the training/eval/
calibration/inference pipeline without needing real photos.

The images have no biological meaning — they're solid-color-plus-noise
PNGs, one rough color per class, just enough pixel variation for the model
to learn *something* non-degenerate. This only proves the pipeline runs
end-to-end (shapes, losses, checkpoint save/load, API serialization); it
says nothing about real-world accuracy.

Usage:
    python scripts/make_smoke_dataset.py --out /tmp/smoke_dataset --per-class 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402

# Rough base color per raw class, chosen loosely to match the labeling guide
# (green=healthy, brown=decay/dry, etc.) — not meant to be realistic.
BASE_COLORS = {
    "Healthy": (60, 160, 70),
    "Moderate": (150, 180, 90),
    "Low": (170, 150, 110),
    "Decay": (110, 80, 50),
    "Dried": (210, 205, 190),
    "Disease": (140, 100, 120),
}

SPLIT_COUNTS = {"train": 1.0, "validation": 0.3, "test": 0.3}  # relative to --per-class


def make_image(base_color: tuple[int, int, int], size: int, rng: np.random.Generator) -> Image.Image:
    noise = rng.integers(-25, 25, size=(size, size, 3))
    array = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output dataset root (gets <species_slug>/ inside it)")
    parser.add_argument("--per-class", type=int, default=12, help="Images per class in the train split")
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)

    for split, ratio in SPLIT_COUNTS.items():
        n = max(int(args.per_class * ratio), 3)
        for class_name in config.class_names:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            base_color = BASE_COLORS[class_name]
            for i in range(n):
                image = make_image(base_color, config.image_size, rng)
                image.save(class_dir / f"{class_name.lower()}_{i:03d}.png")
        print(f"{split}: {n} images/class")

    print(f"Synthetic smoke dataset written to {root.resolve()}")


if __name__ == "__main__":
    main()
