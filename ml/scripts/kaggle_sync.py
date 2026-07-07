"""Sync the labeled dataset with a Kaggle Dataset instead of committing
images to git. This lets the dataset grow to thousands of images without
bloating the repo, while still giving it versioning (Kaggle keeps a version
history) and a shareable URL.

One-time setup:
  1. Create a Kaggle account, then go to kaggle.com/settings -> API ->
     "Create New Token" and copy the username/key token shown there.
  2. Set them as environment variables:
       export KAGGLE_USERNAME=your_username
       export KAGGLE_KEY=your_key
     (If your account instead downloads a kaggle.json file, place it at
     ~/.kaggle/kaggle.json with `chmod 600` — either method works.)
  3. pip install -r requirements.txt (includes the `kaggle` package)

First time only (creates the Kaggle dataset from what's currently in
ml/dataset/):
    cp ml/dataset/dataset-metadata.json.example ml/dataset/dataset-metadata.json
    # edit "id" in that file to "<your-kaggle-username>/mantis-vision-kappaphycus-health"
    python scripts/kaggle_sync.py init

Every time after that (push a new version once you've added more labeled
photos to ml/dataset/):
    python scripts/kaggle_sync.py push -m "Add 50 more Disease examples"

On a fresh machine / training environment, pull the dataset down instead of
re-labeling anything:
    python scripts/kaggle_sync.py download --dataset <your-kaggle-username>/mantis-vision-kappaphycus-health
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402

METADATA_PATH = config.dataset_dir / "dataset-metadata.json"


def _get_api():
    # Imported lazily so `python scripts/kaggle_sync.py --help` doesn't
    # require kaggle.json to already exist.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _require_metadata() -> str:
    if not METADATA_PATH.exists():
        raise SystemExit(
            f"Missing {METADATA_PATH}.\n"
            f"Copy {METADATA_PATH.name}.example to {METADATA_PATH.name}, "
            'set "id" to <your-kaggle-username>/<dataset-slug>, then re-run.'
        )
    metadata = json.loads(METADATA_PATH.read_text())
    dataset_id = metadata.get("id", "")
    if not dataset_id or dataset_id.startswith("YOUR_KAGGLE_USERNAME"):
        raise SystemExit(f'Edit "id" in {METADATA_PATH} to your real Kaggle username/slug first.')
    return dataset_id


def init() -> None:
    dataset_id = _require_metadata()
    api = _get_api()
    api.dataset_create_new(str(config.dataset_dir), dir_mode="zip", quiet=False)
    print(f"Created {dataset_id} -> https://www.kaggle.com/datasets/{dataset_id}")


def push(message: str) -> None:
    dataset_id = _require_metadata()
    api = _get_api()
    api.dataset_create_version(str(config.dataset_dir), version_notes=message, dir_mode="zip", quiet=False)
    print(f"Pushed new version of {dataset_id} -> https://www.kaggle.com/datasets/{dataset_id}")


def download(dataset: str) -> None:
    if not dataset or "/" not in dataset or dataset.upper().startswith("YOUR_"):
        raise SystemExit(
            f'"{dataset}" doesn\'t look like a real dataset id.\n'
            "Set --dataset to <your-kaggle-username>/<dataset-slug>, copied from your "
            "dataset's actual URL on kaggle.com/datasets/... — not the placeholder text."
        )
    api = _get_api()
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(config.dataset_dir), unzip=True, quiet=False)
    print(f"Downloaded {dataset} -> {config.dataset_dir}")
    print("Run `python -m src.data.validate_dataset` next.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the Kaggle dataset for the first time")

    push_parser = subparsers.add_parser("push", help="Push ml/dataset/ to Kaggle as a new version")
    push_parser.add_argument("-m", "--message", required=True, help="Version notes")

    download_parser = subparsers.add_parser("download", help="Pull a Kaggle dataset into ml/dataset/")
    download_parser.add_argument("--dataset", required=True, help="<kaggle-username>/<dataset-slug>")

    args = parser.parse_args()
    if args.command == "init":
        init()
    elif args.command == "push":
        push(args.message)
    elif args.command == "download":
        download(args.dataset)


if __name__ == "__main__":
    main()
