"""One-time utility: split a flat labeled folder into train/validation/test.

Input layout (e.g. after bulk-labeling raw photos):
    raw/Healthy/*.jpg
    raw/Moderate/*.jpg
    ...

Output: copies files into ml/dataset/{train,validation,test}/<Class>/ using
the 70/15/15 split from the spec, with a fixed seed for reproducibility.

Usage:
    python scripts/split_dataset.py --source /path/to/raw
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def split_class(files: list[Path], seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["validation"])

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder containing <ClassName>/*.jpg subfolders")
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    args = parser.parse_args()

    source_dir = Path(args.source)
    transfer = shutil.move if args.move else shutil.copy2

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in config.class_names:
            print(f"Skipping unrecognized class folder: {class_name}")
            continue

        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not files:
            print(f"No images found for {class_name}, skipping.")
            continue

        splits = split_class(files, args.seed)
        for split_name, split_files in splits.items():
            dest_dir = {
                "train": config.train_dir,
                "validation": config.val_dir,
                "test": config.test_dir,
            }[split_name] / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                transfer(str(f), str(dest_dir / f.name))

        print(
            f"{class_name}: {len(files)} images -> "
            f"train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()
