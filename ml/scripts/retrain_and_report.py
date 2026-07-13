"""Assemble the admin-labeled dataset staged on GitHub, run training and
evaluation, archive the merged dataset to Kaggle, publish the resulting
checkpoint, and report results back into Supabase's `model_runs` table for an
admin to review before promoting.

Orchestrates existing pipeline pieces (src.train, src.evaluate,
scripts.split_dataset.split_class) rather than reimplementing them.

The dataset itself lives on Kaggle (see scripts/kaggle_sync.py), not
Supabase. Admin uploads (apps/web/src/app/api/admin/dataset/route.ts) stage
new labeled photos on a dedicated `dataset-staging` git branch instead of
Supabase Storage, since Kaggle's API only supports syncing a whole local
folder as a new dataset version — not real-time single-file inserts from a
web request. This script is what turns that staging area into a real Kaggle
dataset version: it reads the staged photos from a local checkout of that
branch (.github/workflows/retrain.yml checks it out alongside `main`),
merges them into the working dataset tree, pushes the result to Kaggle, then
resets the staging branch back to `main`'s tip so already-processed uploads
aren't reprocessed next time.

`model_runs` (run status/metrics/checkpoint tracking) stays in Supabase —
that's operational bookkeeping for the admin panel, not part of the dataset
itself, so it's unaffected by this move.

Invoked by .github/workflows/retrain.yml, itself triggered from the admin
panel (apps/web/src/app/api/admin/retrain/route.ts). Never run automatically
on a schedule — retraining here is always a deliberate, admin-initiated action.

Usage:
    python scripts/retrain_and_report.py --model-run-id <uuid> --staging-dir <path>

Required environment variables:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  - service-role Supabase access,
                                                for model_runs bookkeeping only
    GITHUB_TOKEN, GITHUB_REPOSITORY          - to publish the checkpoint as a
                                                GitHub Release asset and reset
                                                the dataset-staging branch
Optional:
    KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_DATASET - pull the archived dataset
        from Kaggle first, then push the merged result back as a new version.
        Skipped (with a warning) if unset — the dataset then only exists in
        this run's ephemeral workspace, so setting these is expected in
        normal operation, not truly optional.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from scripts.split_dataset import split_class  # noqa: E402

REQUEST_TIMEOUT_S = 60
STAGING_BRANCH = "dataset-staging"


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


def maybe_pull_kaggle_archive() -> None:
    # Best-effort: a missing/misconfigured archive shouldn't block retraining
    # on whatever's newly staged, but in normal operation this is how prior
    # runs' images come back into the dataset (they're archived to Kaggle,
    # not kept anywhere else once the staging branch is cleared).
    dataset_id = os.environ.get("KAGGLE_DATASET")
    if not dataset_id or not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print(
            "WARNING: KAGGLE_DATASET/KAGGLE_USERNAME/KAGGLE_KEY not fully set — "
            "skipping Kaggle archive pull. The dataset now only contains this "
            "run's newly staged images, not any prior run's."
        )
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.dataset_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=str(config.dataset_dir), unzip=True, quiet=False)
        print(f"Pulled archived dataset {dataset_id} into {config.dataset_dir}.")
    except Exception as e:  # noqa: BLE001 - archive pull is best-effort, not fatal
        print(f"Could not pull Kaggle archive ({e}); continuing with newly staged images only.")


def push_to_kaggle_archive(message: str) -> None:
    # This is now the dataset's actual durable store (not just a backup), so
    # a failure here is logged loudly — but still doesn't abort the run,
    # since the checkpoint is already trained and worth reporting either way.
    dataset_id = os.environ.get("KAGGLE_DATASET")
    if not dataset_id or not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print(
            "WARNING: KAGGLE_DATASET/KAGGLE_USERNAME/KAGGLE_KEY not fully set — "
            "skipping Kaggle push. The newly staged images are NOT durably "
            "archived anywhere after this run."
        )
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_create_version(str(config.dataset_dir), version_notes=message, dir_mode="zip", quiet=False)
        print(f"Pushed new version of {dataset_id} -> https://www.kaggle.com/datasets/{dataset_id}")
    except Exception as e:  # noqa: BLE001 - log and continue; the checkpoint is still valid
        print(f"WARNING: could not push merged dataset to Kaggle ({e}).")


def materialize_from_staging(staging_dir: Path) -> int:
    """staging_dir is a local checkout of the dataset-staging branch's
    ml/dataset_incoming/ — one subfolder per class (same naming convention as
    the real dataset), containing newly admin-labeled, not-yet-trained-on
    photos. Splits each class's staged images into train/validation/test via
    the same split_class used by scripts/split_dataset.py."""
    if not staging_dir.is_dir():
        print(f"No staging directory at {staging_dir}; nothing new to materialize.")
        return 0

    total = 0
    for class_dir in sorted(p for p in staging_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        if not files:
            continue
        splits = split_class(files, config.seed)
        for split_name, split_files in splits.items():
            dest_dir = {
                "train": config.train_dir,
                "validation": config.val_dir,
                "test": config.test_dir,
            }[split_name] / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                f.rename(dest_dir / f.name)
        total += len(files)
        print(f"{class_dir.name}: added {len(files)} newly labeled images from staging.")
    return total


def clear_staging_branch() -> None:
    """Reset dataset-staging back to main's current tip now that its pending
    uploads have been materialized and archived to Kaggle. Not fatal if it
    fails — the branch just accumulates until the next successful clear."""
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    try:
        main_ref_req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/git/ref/heads/main", headers=headers
        )
        with urllib.request.urlopen(main_ref_req, timeout=REQUEST_TIMEOUT_S) as response:
            main_sha = json.loads(response.read())["object"]["sha"]

        reset_req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/git/refs/heads/{STAGING_BRANCH}",
            data=json.dumps({"sha": main_sha, "force": True}).encode(),
            method="PATCH",
            headers={**headers, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(reset_req, timeout=REQUEST_TIMEOUT_S) as response:
            response.read()
        print(f"Cleared {STAGING_BRANCH} back to main's tip.")
    except urllib.error.HTTPError as e:
        if e.code == 422:
            print(f"{STAGING_BRANCH} branch doesn't exist yet; nothing to clear.")
        else:
            print(f"WARNING: could not clear {STAGING_BRANCH} ({e}); it will be reprocessed next run.")


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


def refresh_label_map() -> None:
    # Refresh the human-auditable folder -> integer-ID map from what's now on
    # disk, so it always reflects the classes this run actually trains on.
    from scripts.build_label_map import COLUMNS, scan_folders

    rows = scan_folders(config.train_dir)
    out_path = config.metadata_dir / "label_map.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument(
        "--staging-dir",
        required=True,
        help="Local checkout of the dataset-staging branch's ml/dataset_incoming/ directory",
    )
    args = parser.parse_args()
    model_run_id = args.model_run_id

    try:
        update_run(model_run_id, status="running")

        maybe_pull_kaggle_archive()

        new_count = materialize_from_staging(Path(args.staging_dir))

        refresh_label_map()

        # The model needs every condition represented in the train split — an
        # empty condition class would crash training on an empty ImageFolder
        # class and make the confusion matrix meaningless.
        from src.data.validate_dataset import condition_counts

        counts = condition_counts(config.train_dir)
        empty = [name for name, count in counts.items() if count == 0]
        if empty:
            raise RuntimeError(
                f"Cannot train: no training images for condition(s) {empty}. "
                "Label at least one image per condition (including Background) first."
            )

        from src.evaluate import evaluate
        from src.train import train

        train()
        results = evaluate()

        checkpoint_path = config.checkpoints_dir / "best_model.pt"
        checkpoint_url = create_github_release(model_run_id, checkpoint_path)

        push_to_kaggle_archive(f"Retraining run {model_run_id}: +{new_count} newly labeled images.")
        clear_staging_branch()

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
