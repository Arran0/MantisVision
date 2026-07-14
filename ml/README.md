---
title: Mantis Vision Inference API
emoji: 👁️
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Mantis Vision — ML Pipeline

Schema-driven multi-head seaweed analyzer. A shared EfficientNet-B0 backbone
grows one head per measurement defined in the active **measurement schema** —
classification (e.g. condition, disease subtype), regression (e.g. health
score, dried/decayed extent), or segmentation (e.g. biofouling coverage).
Adding a measurement is an admin-UI edit (`/admin/schema`), not a code
change. See [`../docs/STEP_BY_STEP.md`](../docs/STEP_BY_STEP.md) for the full
walkthrough; this file is the quick reference once you already know the
pipeline.

The YAML block above is Hugging Face Spaces' config header — when this
folder is deployed as a Space (see
[`../docs/DEPLOY_HUGGINGFACE.md`](../docs/DEPLOY_HUGGINGFACE.md)), it tells
the platform to build `Dockerfile` and route traffic to port 8000. It's
inert everywhere else (GitHub just renders it as a normal README).

## Layout

```
ml/
  config.py              # paths, hyperparams, and the fallback DEFAULT_SCHEMA
  metadata/schema.json    # the active measurement schema, exported from Supabase
                          # (see scripts/export_schema.py) — absent locally is fine,
                          # config.py falls back to DEFAULT_SCHEMA
  dataset/<species_slug>/{train,validation,test}/
                          # images/, masks/<measurement_key>/, annotations.jsonl
                          # (per-image column annotations — no class folders)
                          # lives on Kaggle, gitignored here
  checkpoints/             # saved model weights (gitignored)
  reports/                 # evaluation outputs: confusion matrix, metrics json (gitignored)
  logs/                    # run logs (gitignored)
  src/
    data/
      annotations.py       # annotations.jsonl parsing -> per-measurement (target, mask)
      dataset.py            # AnnotatedDataset + dataloaders, schema-driven
      transforms.py         # augmentation pipeline
      validate_dataset.py    # manifest sanity-check CLI
    models/efficientnet.py # build_model(schema) grows heads dynamically; checkpoint save/load
    losses.py               # masked multi-task loss, one term per measurement
    train.py                 # two-phase training (frozen heads -> fine-tune) with early stopping
    evaluate.py               # per-measurement metrics: classification report/confusion,
                              # regression MAE, segmentation IoU
    gradcam.py                 # Grad-CAM explainability (targets the primary classification head)
    inference/predictor.py      # schema-driven prediction -> structured per-measurement report
    api/main.py                   # FastAPI inference service (+ hot-swap /admin/reload)
  scripts/
    export_schema.py         # pulls the active measurement_schema from Supabase -> metadata/schema.json
    build_label_map.py         # manifest -> metadata/label_map.csv (human-auditable training audit trail)
    split_dataset.py             # generic 70/15/15 list splitter (used by retrain_and_report.py)
    kaggle_sync.py                 # push/pull the labeled dataset to/from a Kaggle Dataset
    export_model.py                 # checkpoint -> multi-output ONNX for web/mobile deployment
    retrain_and_report.py             # CI retraining orchestration (admin "Retrain" button)
```

## Quick commands

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. get the active schema + labeled photos
python scripts/export_schema.py                       # -> metadata/schema.json (needs SUPABASE_URL/SERVICE_ROLE_KEY)
python scripts/kaggle_sync.py download --dataset <username>/mantis-vision-kappaphycus-health
python scripts/build_label_map.py                      # regenerate metadata/label_map.csv
python -m src.data.validate_dataset                    # sanity check before training

# 2. train
python -m src.train

# 3. evaluate on the held-out test split
python -m src.evaluate

# 4. explain a single prediction
python -m src.gradcam path/to/photo.jpg

# 5. serve predictions
uvicorn src.api.main:app --reload --port 8000

# 6. export for web/mobile
python scripts/export_model.py
```

The measurement schema (species, measurements, classes, display thresholds)
is admin-editable at `/admin/schema` and versioned in Supabase — see
[`../docs/DATASET_LABELING_GUIDE.md`](../docs/DATASET_LABELING_GUIDE.md) for
how per-image annotations are structured and labeled.
