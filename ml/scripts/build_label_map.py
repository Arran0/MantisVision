"""Scan the dataset's class folders and write a human-auditable label map:
folder name -> integer condition/subtype IDs plus the derived level and the
heuristic anchors each folder implies.

This is the CSV the labeling workflow refers to when expanding to more
species/diseases — it makes the folder -> integer-ID mapping (which the model
actually trains on) explicit and reviewable, rather than buried in code.

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
from config import config  # noqa: E402
from src.data.labels import derive_targets, health_level, parse_class_folder  # noqa: E402

COLUMNS = [
    "folder_name",
    "condition",
    "condition_id",
    "severity",
    "subtype",
    "subtype_id",
    "disease_name",
    "health_level",
    "health_score_anchor",
    "dried_extent_anchor",
    "decayed_extent_anchor",
]


def scan_folders(split_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        parsed = parse_class_folder(class_dir.name, config.species_slug)
        targets = derive_targets(parsed)
        rows.append(
            {
                "folder_name": class_dir.name,
                "condition": parsed.condition,
                "condition_id": targets["condition_id"],
                "severity": parsed.severity or "",
                "subtype": parsed.subtype or "",
                "subtype_id": targets["subtype_id"] if parsed.condition == "Disease" else "",
                "disease_name": parsed.disease_name or "",
                "health_level": health_level(parsed.condition, parsed.severity) or "",
                "health_score_anchor": targets["health_score"],
                "dried_extent_anchor": targets["dried_extent"],
                "decayed_extent_anchor": targets["decayed_extent"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out", default=None, help="Output CSV (default: metadata/label_map.csv)")
    args = parser.parse_args()

    split_dir = config.dataset_dir / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"No such split directory: {split_dir}")

    rows = scan_folders(split_dir)
    if not rows:
        raise SystemExit(f"No class folders found under {split_dir}.")

    out_path = Path(args.out) if args.out else (config.metadata_dir / "label_map.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} class folders -> {out_path}")


if __name__ == "__main__":
    main()
