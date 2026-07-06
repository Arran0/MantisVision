"""Shared logging setup so every script writes to console + a run-scoped file."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(logs_dir / f"{name}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
