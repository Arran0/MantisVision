"""Pulls the active measurement_schema document from Supabase and writes it to
ml/metadata/schema.json, so config.py's load_schema() (called at import time)
picks it up for training/serving.

Run as its own step in .github/workflows/retrain.yml, BEFORE the retraining
step, so the retraining process starts fresh with schema.json already on
disk (config.SCHEMA is loaded once at import time — writing the file from
*within* the same process that already imported config would be too late for
that process's own global, though retrain_and_report.py additionally applies
the freshly-fetched schema to config.SCHEMA directly for its own run, so it
doesn't strictly depend on running as a prior step; see its own docstring).

Usage:
    python scripts/export_schema.py [--out PATH]

Required environment variables:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config, schema_to_dict, DEFAULT_SCHEMA  # noqa: E402

REQUEST_TIMEOUT_S = 60


def fetch_active_schema() -> dict:
    """Returns the active measurement_schema doc, or DEFAULT_SCHEMA (as a
    dict) if no row exists yet. Network/auth failures are NOT swallowed —
    unlike the Kaggle archive pull, a schema fetch failure means we can't be
    sure what the model should train on, so the caller should abort the run
    rather than silently training on a possibly-wrong fallback."""
    # .strip() guards against a stray trailing newline/whitespace in the
    # secret value (e.g. from a copy-paste into the GitHub secret field) —
    # that would otherwise reach http.client as a literal "\n" in the
    # Authorization header value and raise "Invalid header value".
    url = os.environ["SUPABASE_URL"].strip().rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()
    request = urllib.request.Request(
        f"{url}/rest/v1/measurement_schema?select=doc&order=created_at.desc&limit=1",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        rows = json.loads(response.read())
    if not rows:
        return schema_to_dict(DEFAULT_SCHEMA)
    return rows[0]["doc"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=config.metadata_dir / "schema.json")
    args = parser.parse_args()

    doc = fetch_active_schema()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Exported active measurement schema ({len(doc.get('measurements', []))} measurements) -> {args.out}")


if __name__ == "__main__":
    main()
