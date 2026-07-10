"""Assemble the current admin-labeled dataset from Supabase, run training and
evaluation, publish the resulting checkpoint, and report results back into
the `model_runs` table for an admin to review before promoting.

Orchestrates existing pipeline pieces (src.train, src.evaluate,
scripts.split_dataset.split_class) rather than reimplementing them, and talks
to Supabase's PostgREST/Storage HTTP APIs directly via urllib — this repo has
no Supabase Python SDK dependency, matching the style already used in
src/api/main.py's checkpoint download.

Invoked by .github/workflows/retrain.yml, itself triggered from the admin
panel (apps/web/src/app/api/admin/retrain/route.ts). Never run automatically
on a schedule — retraining here is always a deliberate, admin-initiated action.

Usage:
    python scripts/retrain_and_report.py --model-run-id <uuid>

Required environment variables:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  - service-role Supabase access
    GITHUB_TOKEN, GITHUB_REPOSITORY          - to publish the checkpoint as a
                                                GitHub Release asset
Optional:
    KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_DATASET - pull the archived dataset
        from Kaggle first, before layering the new admin-labeled images on
        top (see scripts/kaggle_sync.py). Skipped if unset.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from scripts.split_dataset import split_class  # noqa: E402

REQUEST_TIMEOUT_S = 60


def _supabase_env() -> tuple[str, str]:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return url, key


def _supabase_request(method: str, path: str, *, body: dict | None = None, params: str = "") -> object:
    url, key = _supabase_env()
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{url}/rest/v1/{path}{params}",
        data=data,
        method=method,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def update_run(model_run_id: str, **fields) -> None:
    _supabase_request("PATCH", "model_runs", body=fields, params=f"?id=eq.{model_run_id}")


def fetch_labeled_images() -> list[dict]:
    result = _supabase_request(
        "GET", "training_images", params="?status=eq.labeled&select=id,storage_path,health"
    )
    return result or []


def download_storage_object(storage_path: str, dest: Path) -> None:
    url, key = _supabase_env()
    request = urllib.request.Request(
        f"{url}/storage/v1/object/training-images/{storage_path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response, open(dest, "wb") as f:
        f.write(response.read())


def maybe_pull_kaggle_archive() -> None:
    # Best-effort: the durable dataset archive is a nice-to-have base to
    # train on top of, but a missing/misconfigured archive shouldn't block
    # retraining on the newly labeled images alone.
    dataset_id = os.environ.get("KAGGLE_DATASET")
    if not dataset_id or not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print("Skipping Kaggle archive pull (KAGGLE_DATASET/KAGGLE_USERNAME/KAGGLE_KEY not fully set).")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.dataset_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=str(config.dataset_dir), unzip=True, quiet=False)
        print(f"Pulled archived dataset {dataset_id} into {config.dataset_dir}.")
    except Exception as e:  # noqa: BLE001 - archive pull is best-effort, not fatal
        print(f"Could not pull Kaggle archive ({e}); continuing with newly labeled images only.")


def materialize_new_labels(images: list[dict], raw_dir: Path) -> int:
    by_class: dict[str, list[Path]] = {}
    for image in images:
        health = image["health"]
        if health not in config.class_names:
            print(f"Skipping training_images row {image['id']}: unrecognized class {health!r}.")
            continue
        ext = Path(image["storage_path"]).suffix or ".jpg"
        dest = raw_dir / health / f"{image['id']}{ext}"
        download_storage_object(image["storage_path"], dest)
        by_class.setdefault(health, []).append(dest)

    total = 0
    for class_name, files in by_class.items():
        splits = split_class(sorted(files), config.seed)
        for split_name, split_files in splits.items():
            dest_dir = {
                "train": config.train_dir,
                "validation": config.val_dir,
                "test": config.test_dir,
            }[split_name] / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                f.rename(dest_dir / f.name)
        total += len(files)
        print(f"{class_name}: added {len(files)} newly labeled images.")
    return total


def create_github_release(model_run_id: str, checkpoint_path: Path) -> str:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    tag = f"model-run-{model_run_id}"

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases",
        data=json.dumps(
            {"tag_name": tag, "name": tag, "body": f"Automated retraining run {model_run_id}."}
        ).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        release = json.loads(response.read())

    upload_url = release["upload_url"].split("{")[0]
    content_type = mimetypes.guess_type(str(checkpoint_path))[0] or "application/octet-stream"
    upload_request = urllib.request.Request(
        f"{upload_url}?name={checkpoint_path.name}",
        data=checkpoint_path.read_bytes(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": content_type,
        },
    )
    with urllib.request.urlopen(upload_request, timeout=REQUEST_TIMEOUT_S) as response:
        asset = json.loads(response.read())

    return asset["browser_download_url"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run-id", required=True)
    args = parser.parse_args()
    model_run_id = args.model_run_id

    try:
        update_run(model_run_id, status="running")

        maybe_pull_kaggle_archive()

        images = fetch_labeled_images()
        raw_dir = config.dataset_dir / "_incoming"
        new_count = materialize_new_labels(images, raw_dir)

        from src.data.validate_dataset import validate_split

        train_counts = validate_split(config.train_dir, config.class_names)
        empty_classes = [name for name, count in train_counts.items() if count == 0]
        if empty_classes:
            raise RuntimeError(
                f"Cannot train: no training images for class(es) {empty_classes}. "
                "Label at least one image per class first."
            )

        from src.evaluate import evaluate
        from src.train import train

        train()
        results = evaluate()

        checkpoint_path = config.checkpoints_dir / "best_model.pt"
        checkpoint_url = create_github_release(model_run_id, checkpoint_path)

        update_run(
            model_run_id,
            status="completed",
            metrics=results,
            checkpoint_url=checkpoint_url,
            dataset_image_count=new_count,
            github_run_id=os.environ.get("GITHUB_RUN_ID"),
        )
        print(f"Retraining run {model_run_id} completed -> {checkpoint_url}")
    except Exception as e:  # noqa: BLE001 - must always report failure back to model_runs
        traceback.print_exc()
        update_run(model_run_id, status="failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
