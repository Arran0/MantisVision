"""Scan a split's annotations.jsonl manifest and write a human-auditable
label map: one row per image, showing exactly what each schema measurement
resolves to for training (blank when that measurement is masked out — i.e.
missing, or its applies_when isn't satisfied by that image's other values).

This is the audit trail for the column/CSV annotation model (replacing the
old folder-name -> integer-ID mapping, which no longer exists now that
class folders don't either) — it makes what the model actually trains on
explicit and reviewable, rather than buried in a JSONL file.

Usage:
    python scripts/build_label_map.py           # scans dataset/<slug>/train
    python scripts/build_label_map.py --split validation
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Schema, config  # noqa: E402
from src.data.annotations import derive_targets, load_manifest  # noqa: E402


def _fieldnames(schema: Schema) -> list[str]:
    fields = ["filename"]
    for m in schema.measurements:
        if m.type == "classification":
            fields += [m.key, f"{m.key}_id"]
        else:  # regression or segmentation
            fields.append(m.key)
    return fields


def scan_manifest(split_dir: Path, schema: Schema) -> list[dict]:
    manifest_path = split_dir / "annotations.jsonl"
    if not manifest_path.exists():
        return []

    rows: list[dict] = []
    for row in load_manifest(manifest_path):
        targets = derive_targets(schema, row.measurements)
        record: dict = {"filename": row.filename}
        for m in schema.measurements:
            if m.type == "classification":
                masked_in = targets[f"{m.key}_mask"] > 0
                record[m.key] = row.measurements.get(m.key, "") if masked_in else ""
                record[f"{m.key}_id"] = targets[f"{m.key}_id"] if masked_in else ""
            elif m.type == "regression":
                record[m.key] = targets[m.key] if targets[f"{m.key}_mask"] > 0 else ""
            elif m.type == "segmentation":
                record[m.key] = row.masks.get(m.key, "")
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out", default=None, help="Output CSV (default: metadata/label_map.csv)")
    args = parser.parse_args()

    split_dir = {"train": config.train_dir, "validation": config.val_dir, "test": config.test_dir}[args.split]
    schema = config.SCHEMA

    rows = scan_manifest(split_dir, schema)
    if not rows:
        raise SystemExit(f"No annotations.jsonl (or it's empty) under {split_dir}.")

    out_path = Path(args.out) if args.out else (config.metadata_dir / "label_map.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(schema))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} images -> {out_path}")


if __name__ == "__main__":
    main()
