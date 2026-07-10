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

Multi-head seaweed analyzer for *Kappaphycus alvarezii*. A shared
EfficientNet-B0 backbone feeds five heads: condition (incl. a Background
"no seaweed" class), a 0–100 health-score regression, a masked disease
subtype, and dried/decayed extent regression. See
[`../docs/STEP_BY_STEP.md`](../docs/STEP_BY_STEP.md) for the full walkthrough;
this file is the quick reference once you already know the pipeline.

The YAML block above is Hugging Face Spaces' config header — when this
folder is deployed as a Space (see
[`../docs/DEPLOY_HUGGINGFACE.md`](../docs/DEPLOY_HUGGINGFACE.md)), it tells
the platform to build `Dockerfile` and route traffic to port 8000. It's
inert everywhere else (GitHub just renders it as a normal README).

## Layout

```
ml/
  config.py              # single source of truth: species, taxonomy, anchors, hyperparams
  dataset/<slug>/         # {train,validation,test}/<class_folder>/ — lives on Kaggle, gitignored here
  checkpoints/             # saved model weights (gitignored)
  reports/                 # evaluation outputs: confusion matrix, metrics json (gitignored)
  logs/                    # run logs (gitignored)
  metadata/labels_template.csv, label_map.csv (generated)
  src/
    data/                  # transforms, labels.py (folder-name parser), SeaweedDataset, validation CLI
    models/efficientnet.py # multi-head model + checkpoint save/load
    losses.py              # multi-task masked loss shared by train/evaluate
    train.py               # two-phase training (frozen heads -> fine-tune) with early stopping
    evaluate.py            # per-head metrics: condition report/confusion + subtype + regression MAE
    gradcam.py             # Grad-CAM explainability (targets the condition head)
    inference/             # predictor.py (full output) + explanations.py (text templates)
    api/main.py             # FastAPI inference service
  scripts/
    split_dataset.py        # flat labeled folder -> 70/15/15 train/val/test split
    build_label_map.py      # scan dataset -> metadata/label_map.csv (folder -> integer IDs)
    kaggle_sync.py           # push/pull the labeled dataset to/from a Kaggle Dataset
    export_model.py         # checkpoint -> multi-output ONNX for web/mobile deployment
    retrain_and_report.py   # CI retraining orchestration (admin "Retrain" button)
```

## Quick commands

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. get labeled photos into dataset/<slug>/{train,validation,test}/<class_folder>/
python scripts/kaggle_sync.py download --dataset <username>/mantis-vision-kappaphycus-health
python scripts/build_label_map.py        # regenerate metadata/label_map.csv
python -m src.data.validate_dataset      # sanity check before training

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

Conditions: `Healthy`, `Disease`, `Decay`, `Dried`, `Background`. Class-folder
names encode structured labels (e.g. `<slug>_Disease_Moderate_IceIce`) — the
naming convention and labeling standards are in
[`../docs/DATASET_LABELING_GUIDE.md`](../docs/DATASET_LABELING_GUIDE.md).
