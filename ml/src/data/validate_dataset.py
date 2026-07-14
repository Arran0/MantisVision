"""CLI: sanity-check a split's annotations.jsonl manifest before training.

Checks:
  - every image file referenced in the manifest opens correctly (catches
    corrupt/truncated files)
  - the schema's primary classification's background_class is represented in
    each split (the model needs negatives to avoid false positives)
  - reports per-measurement counts: per-class counts for classifications,
    non-null value counts for regressions, mask-present counts for
    segmentations — and flags severe class imbalance on the primary
    classification

Usage:
    python -m src.data.validate_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Schema, config  # noqa: E402
from src.data.annotations import AnnotationRow, load_manifest  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _check_images(split_dir: Path, rows: list[AnnotationRow]) -> int:
    valid = 0
    for row in rows:
        path = split_dir / "images" / row.filename
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            print(f"  [CORRUPT or MISSING] {path}")
    return valid


def measurement_counts(split_dir: Path, schema: Schema) -> dict[str, dict[str, int]]:
    """Per-measurement counts for a split: classification -> {class_name:
    count}; regression -> {"has_value": count, "missing": count};
    segmentation -> {"has_mask": count, "missing": count}."""
    manifest_path = split_dir / "annotations.jsonl"
    rows = load_manifest(manifest_path) if manifest_path.exists() else []

    counts: dict[str, dict[str, int]] = {}
    for m in schema.measurements:
        if m.type == "classification":
            counts[m.key] = {c.name: 0 for c in m.classes}
        elif m.type == "regression":
            counts[m.key] = {"has_value": 0, "missing": 0}
        elif m.type == "segmentation":
            counts[m.key] = {"has_mask": 0, "missing": 0}

    for row in rows:
        for m in schema.measurements:
            if m.type == "classification":
                value = row.measurements.get(m.key)
                if value in counts[m.key]:
                    counts[m.key][value] += 1
            elif m.type == "regression":
                key = "has_value" if isinstance(row.measurements.get(m.key), (int, float)) else "missing"
                counts[m.key][key] += 1
            elif m.type == "segmentation":
                key = "has_mask" if m.key in row.masks else "missing"
                counts[m.key][key] += 1

    return counts


def main() -> None:
    schema = config.SCHEMA
    primary = schema.primary_classification()
    splits = {"train": config.train_dir, "validation": config.val_dir, "test": config.test_dir}

    all_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split_name, split_dir in splits.items():
        print(f"\n== {split_name} ({split_dir}) ==")
        manifest_path = split_dir / "annotations.jsonl"
        if not manifest_path.exists():
            print(f"  [MISSING] {manifest_path}")
            all_counts[split_name] = measurement_counts(split_dir, schema)
            continue

        rows = load_manifest(manifest_path)
        valid = _check_images(split_dir, rows)
        print(f"  {valid}/{len(rows)} images verified OK")

        counts = measurement_counts(split_dir, schema)
        all_counts[split_name] = counts
        if primary is not None and counts.get(primary.key, {}).get(primary.background_class, 0) == 0:
            print(
                f"  [WARN] no {primary.background_class!r} sample for {primary.key!r} — "
                "the model needs negatives to avoid false positives"
            )

    if primary is not None:
        print(f"\n== Per-class counts: {primary.label} ({primary.key}) ==")
        header = f"{'Class':<14}" + "".join(f"{s:>12}" for s in splits)
        print(header)
        for name in primary.class_names():
            row = f"{name:<14}" + "".join(f"{all_counts[s][primary.key].get(name, 0):>12}" for s in splits)
            print(row)

        total_train = sum(all_counts["train"].get(primary.key, {}).values())
        if total_train == 0:
            print("\nNo training images found yet — label some photos first.")
            return

        print(f"\n== Class balance ({primary.key}, train split) ==")
        train_counts = all_counts["train"][primary.key]
        max_count = max(train_counts.values()) or 1
        for name, count in train_counts.items():
            ratio = count / max_count if max_count else 0
            flag = "  <-- underrepresented, consider collecting more" if ratio < 0.3 and count > 0 else ""
            print(f"  {name:<14} {count:>6} images ({ratio:.0%} of largest class){flag}")

    for m in schema.measurements:
        if m is primary:
            continue
        print(f"\n== {m.label} ({m.key}, {m.type}) ==")
        header = f"{'':<14}" + "".join(f"{s:>12}" for s in splits)
        print(header)
        for key in all_counts["train"].get(m.key, {}):
            row = f"{key:<14}" + "".join(f"{all_counts[s][m.key].get(key, 0):>12}" for s in splits)
            print(row)


if __name__ == "__main__":
    main()
