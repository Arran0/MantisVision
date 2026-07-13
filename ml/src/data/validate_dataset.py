"""CLI: sanity-check the dataset before training.

Checks:
  - every class folder name parses under the naming convention (labels.py)
  - the Background negative class is present in each split
  - every image file opens correctly (catches corrupt/truncated files)
  - reports per-split, per-condition image counts and flags severe imbalance

Note: multiple class folders can map to the same *condition* (e.g. several
Disease_<Severity>_<Subtype> folders all count as "Disease"), so counts are
aggregated per condition — that's what the model's condition head learns.

Usage:
    python -m src.data.validate_dataset
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config  # noqa: E402
from src.data.labels import BACKGROUND, parse_class_folder  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _count_images(class_dir: Path) -> int:
    files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    valid = 0
    for f in files:
        try:
            with Image.open(f) as img:
                img.verify()
            valid += 1
        except (UnidentifiedImageError, OSError):
            print(f"  [CORRUPT] {f}")
    return valid


def condition_counts(split_dir: Path) -> dict[str, int]:
    """Per-condition valid-image counts for a split, keyed by the fixed
    config.condition_classes (so a condition with zero images shows as 0)."""
    counts: dict[str, int] = {name: 0 for name in config.condition_classes}
    if not split_dir.is_dir():
        return counts
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        parsed = parse_class_folder(class_dir.name, config.species_slug)
        counts[parsed.condition] += _count_images(class_dir)
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
        if not split_dir.is_dir():
            print(f"  [MISSING] {split_dir}")
        elif BACKGROUND not in {p.name for p in split_dir.iterdir() if p.is_dir()}:
            print(f"  [WARN] no {BACKGROUND!r} folder — the model needs negatives to avoid false positives")
        all_counts[split_name] = condition_counts(split_dir)

    print("\n== Per-condition image counts ==")
    header = f"{'Condition':<12}" + "".join(f"{s:>12}" for s in splits)
    print(header)
    for name in config.condition_classes:
        row = f"{name:<12}" + "".join(f"{all_counts[s][name]:>12}" for s in splits)
        print(row)

    total_train = sum(all_counts["train"].values())
    if total_train == 0:
        print("\nNo training images found yet — add photos under the class folders.")
        return

    print("\n== Class balance (train split) ==")
    max_count = max(all_counts["train"].values()) or 1
    for name, count in all_counts["train"].items():
        ratio = count / max_count if max_count else 0
        flag = "  <-- underrepresented, consider collecting more" if ratio < 0.3 and count > 0 else ""
        print(f"  {name:<12} {count:>6} images ({ratio:.0%} of largest condition){flag}")


if __name__ == "__main__":
    main()
