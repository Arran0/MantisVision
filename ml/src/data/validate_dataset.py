"""CLI: sanity-check the dataset before training.

Checks:
  - every expected class folder exists in train/validation/test
  - every image file opens correctly (catches corrupt/truncated files)
  - reports per-split, per-class image counts and flags severe class imbalance

Usage:
    python -m src.data.validate_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_split(split_dir: Path, class_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            print(f"  [MISSING] {class_dir}")
            counts[class_name] = 0
            continue

        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        corrupt = 0
        for f in files:
            try:
                with Image.open(f) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                corrupt += 1
                print(f"  [CORRUPT] {f}")

        counts[class_name] = len(files) - corrupt
    return counts


def main() -> None:
    splits = {
        "train": config.train_dir,
        "validation": config.val_dir,
        "test": config.test_dir,
    }

    all_counts: dict[str, dict[str, int]] = {}
    for split_name, split_dir in splits.items():
        print(f"\n== {split_name} ({split_dir}) ==")
        all_counts[split_name] = validate_split(split_dir, config.class_names)

    print("\n== Per-class image counts ==")
    header = f"{'Class':<12}" + "".join(f"{s:>12}" for s in splits)
    print(header)
    for class_name in config.class_names:
        row = f"{class_name:<12}"
        for split_name in splits:
            row += f"{all_counts[split_name][class_name]:>12}"
        print(row)

    total_train = sum(all_counts["train"].values())
    if total_train == 0:
        print("\nNo training images found yet — add photos under ml/dataset/train/<Class>/")
        return

    print("\n== Class balance (train split) ==")
    max_count = max(all_counts["train"].values()) or 1
    for class_name, count in all_counts["train"].items():
        ratio = count / max_count if max_count else 0
        flag = "  <-- underrepresented, consider collecting more" if ratio < 0.3 and count > 0 else ""
        print(f"  {class_name:<12} {count:>6} images ({ratio:.0%} of largest class){flag}")


if __name__ == "__main__":
    main()
