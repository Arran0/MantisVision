"""A 70/15/15 train/validation/test list splitter, seeded for reproducibility.

Used by scripts/retrain_and_report.py to split newly-labeled rows fetched
from Supabase (there are no class folders to split anymore — annotations are
a per-row column/CSV-style manifest, see src/data/annotations.py).
"""
from __future__ import annotations

import random
from pathlib import Path

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
