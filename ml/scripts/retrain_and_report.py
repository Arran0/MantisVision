"""Assemble the current admin-labeled dataset from Supabase, run training and
evaluation, publish the resulting checkpoint, and report results back into
the `model_runs` table for an admin to review before promoting.

Orchestrates existing pipeline pieces (src.train, src.evaluate) rather than
reimplementing them, and talks to Supabase's PostgREST/Storage HTTP APIs
directly via urllib — this repo has no Supabase Python SDK dependency,
matching the style already used in src/api/main.py's checkpoint download.

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
import config as config_module  # noqa: E402
from config import Schema, config, schema_from_dict  # noqa: E402
from scripts.export_schema import fetch_active_schema  # noqa: E402
from scripts.split_dataset import split_class  # noqa: E402

REQUEST_TIMEOUT_S = 60


def _supabase_env() -> tuple[str, str]:
    # .strip() guards against a stray trailing newline/whitespace in the
    # secret value (e.g. from a copy-paste into the GitHub secret field) —
    # that would otherwise reach http.client as a literal "\n" in the
    # Authorization header value and raise "Invalid header value".
    url = os.environ["SUPABASE_URL"].strip().rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()
    return url, key


def _supabase_request(method: str, path: str, *, body: dict | None = None, params: str = "") -> object:
    url, key = _supabase_env()
    # allow_nan=False turns a stray NaN/Infinity in the body (e.g. an
    # undefined metric that slipped past its own sanitization) into an
    # immediate, readable ValueError here, rather than json.dumps silently
    # emitting the bare (invalid-JSON) token NaN/Infinity that PostgREST
    # then rejects downstream as an opaque "HTTP Error 400: Bad Request".
    data = json.dumps(body, allow_nan=False).encode() if body is not None else None
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
    # Which measurement keys matter is schema-defined now, not a fixed column
    # list, so pull the whole `measurements` map per row instead of naming
    # individual columns. `split` is an admin-chosen train/validation/test
    # pin (see materialize_new_labels) — null means "assign automatically".
    result = _supabase_request(
        "GET",
        "training_images",
        params="?status=eq.labeled&select=id,storage_path,measurements,split",
    )
    return result or []


def download_storage_object(bucket: str, storage_path: str, dest: Path) -> None:
    url, key = _supabase_env()
    request = urllib.request.Request(
        f"{url}/storage/v1/object/{bucket}/{storage_path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response, open(dest, "wb") as f:
        f.write(response.read())


def maybe_pull_kaggle_archive() -> None:
    # Best-effort: the durable dataset archive is a nice-to-have base to
    # train on top of, but a missing/misconfigured archive shouldn't block
    # retraining on the newly labeled images alone.
    #
    # NOTE: a Kaggle archive is assumed to already be laid out as
    # images/ + masks/<key>/ + annotations.jsonl per split (the same
    # manifest convention src.data.dataset.AnnotatedDataset reads) — the old
    # class-folder convention this repo used before the measurement schema
    # existed is no longer understood by the training pipeline. Publish any
    # archive in the new layout going forward.
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


SPLIT_NAMES = ("train", "validation", "test")


def materialize_new_labels(images: list[dict], schema: Schema, raw_dir: Path) -> int:
    """Downloads each labeled row's photo (and any segmentation mask files
    its measurements reference) into a flat staging area, splits the ROWS
    (not folders — there are no class folders anymore) into train/
    validation/test, and appends each split's images/masks/annotations.jsonl.
    A value missing from a row's `measurements` map simply isn't written into
    that row's manifest entry — src.data.annotations.derive_targets treats an
    absent key as "no ground truth yet" and masks it out of that head's loss,
    it does not fabricate a placeholder value.

    A row whose `training_images.split` column is set (an admin pinned it to
    a specific split from the Dataset page's Edit form) goes straight to that
    split; everything else falls back to split_class's random ratio-based
    assignment, applied only across the unpinned remainder."""
    staged: list[dict] = []
    for image in images:
        image_id = image["id"]
        measurements: dict = image.get("measurements") or {}
        ext = Path(image["storage_path"]).suffix or ".jpg"
        filename = f"{image_id}{ext}"
        download_storage_object("training-images", image["storage_path"], raw_dir / "images" / filename)

        row_measurements: dict = {}
        row_masks: dict = {}
        for m in schema.measurements:
            value = measurements.get(m.key)
            if value is None:
                continue
            if m.type == "segmentation":
                mask_ext = Path(str(value)).suffix or ".png"
                mask_filename = f"{image_id}{mask_ext}"
                download_storage_object("training-masks", value, raw_dir / "masks" / m.key / mask_filename)
                row_masks[m.key] = mask_filename
            else:
                row_measurements[m.key] = value

        pinned_split = image.get("split")
        staged.append({
            "filename": filename,
            "measurements": row_measurements,
            "masks": row_masks,
            "pinned_split": pinned_split if pinned_split in SPLIT_NAMES else None,
        })

    pinned: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    unpinned: list[dict] = []
    for row in staged:
        target = row.pop("pinned_split")
        (pinned[target] if target else unpinned).append(row)

    # split_class is a plain list splitter despite its class-folder-oriented
    # name/docstring (scripts/split_dataset.py) — reused here for row dicts.
    auto_splits = split_class(unpinned, config.seed)
    splits = {name: pinned[name] + auto_splits[name] for name in SPLIT_NAMES}

    total = 0
    for split_name, split_rows in splits.items():
        split_dir = {"train": config.train_dir, "validation": config.val_dir, "test": config.test_dir}[split_name]
        (split_dir / "images").mkdir(parents=True, exist_ok=True)

        manifest_lines = []
        for row in split_rows:
            (raw_dir / "images" / row["filename"]).rename(split_dir / "images" / row["filename"])
            for key, mask_filename in row["masks"].items():
                mask_dest_dir = split_dir / "masks" / key
                mask_dest_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "masks" / key / mask_filename).rename(mask_dest_dir / mask_filename)
            manifest_lines.append(
                json.dumps({"filename": row["filename"], "measurements": row["measurements"], "masks": row["masks"]})
            )

        if manifest_lines:
            manifest_path = split_dir / "annotations.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "a") as f:
                f.write("\n".join(manifest_lines) + "\n")
        total += len(split_rows)

    pinned_total = sum(len(rows) for rows in pinned.values())
    counts = ", ".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES)
    pinned_note = f"; {pinned_total} manually pinned" if pinned_total else ""
    print(f"Materialized {total} newly labeled images across train/validation/test ({counts}){pinned_note}.")
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

        # Fetch the schema this run actually trains on (may differ from
        # whatever config.SCHEMA held at import time — e.g. an admin edit
        # since the last export, or a fresh checkout with no schema.json at
        # all) and make it authoritative for the rest of this process, for
        # train()'s default when we don't pass one explicitly.
        schema_doc = fetch_active_schema()
        schema = schema_from_dict(schema_doc)
        config_module.SCHEMA = schema
        config.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(config.metadata_dir / "schema.json", "w") as f:
            json.dump(schema_doc, f, indent=2)
        print(f"Training with schema: {[m.key for m in schema.measurements]}")

        maybe_pull_kaggle_archive()

        images = fetch_labeled_images()
        raw_dir = config.dataset_dir / "_incoming"
        new_count = materialize_new_labels(images, schema, raw_dir)

        # Dataset adequacy (e.g. "is there at least one background sample
        # per split") is now checked inside get_dataloaders (called by
        # train() below) rather than pre-scanned here — that check is
        # schema-driven (src.data.dataset._verify_dataset) and raises the
        # same way a pre-check would, reported to model_runs by the except
        # block below.

        from src.evaluate import evaluate
        from src.train import train

        train(schema=schema)
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
