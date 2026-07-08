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

Multi-head health classifier for *Kappaphycus alvarezii* (Phase 1 / MVP): a
shared EfficientNet-B0 backbone with three heads — health **category**
(Healthy/Moderate/Low), **condition** (None/Dried/Decayed/Diseased, only
meaningful when category != Healthy), and a **health score** regressed to
0–10. See [`../docs/STEP_BY_STEP.md`](../docs/STEP_BY_STEP.md) for the full
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
  config.py              # single source of truth: paths, hyperparams, category/condition/
                          # score-anchor taxonomy (config.CLASS_TARGET_MAP), species slug
  dataset/
    <species_slug>/        # e.g. kappaphycus_alvarezii/ — train/validation/test/<Class>/
                            # images, lives on Kaggle, gitignored here. Species-scoped so a
                            # second species can be added as a sibling folder later.
  checkpoints/             # saved model weights + calibration.json (gitignored)
  reports/                 # evaluation outputs: confusion matrices, score plots, reliability
                            # diagrams, metrics json (gitignored)
  logs/                    # run logs (gitignored)
  metadata/labels_template.csv
  src/
    data/                  # transforms, ImageFolder dataloaders wrapped into 3 training
                            # targets (category/condition/score) via CLASS_TARGET_MAP,
                            # dataset validation CLI
    models/efficientnet.py # multi-head transfer-learning model + checkpoint save/load
    train.py               # two-phase training (frozen heads -> fine-tune) with early stopping
    evaluate.py            # per-head metrics (category/condition/score) + calibration section
    calibration.py         # ECE, temperature scaling, reliability diagrams
    calibrate.py           # CLI: fits + checks confidence calibration, writes calibration.json
    gradcam.py             # Grad-CAM explainability (category head)
    inference/             # predictor.py (full MVP output) + explanations.py (bullet-point
                            # + recommendation templates)
    api/main.py             # FastAPI inference service
  scripts/
    split_dataset.py        # flat labeled folder -> 70/15/15 train/val/test split
    kaggle_sync.py           # push/pull the labeled dataset to/from a Kaggle Dataset
    export_model.py         # checkpoint -> ONNX for web/mobile deployment
    make_smoke_dataset.py   # generates a tiny synthetic dataset to smoke-test the pipeline
```

## Quick commands

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. get labeled photos into dataset/kappaphycus_alvarezii/train|validation|test/<ClassName>/
python scripts/kaggle_sync.py download --dataset <username>/mantis-vision-kappaphycus-health
python -m src.data.validate_dataset      # sanity check before training

# 2. train
python -m src.train

# 3. evaluate on the held-out test split (per-head metrics + raw ECE)
python -m src.evaluate

# 4. fit + check confidence calibration (temperature scaling, ECE before/after)
python -m src.calibrate

# 5. explain a single prediction (Grad-CAM heatmap on the category head)
python -m src.gradcam path/to/photo.jpg

# 6. serve predictions
uvicorn src.api.main:app --reload --port 8000

# 7. export for web/mobile
python scripts/export_model.py
```

Raw class labels (folder names must match exactly): `Healthy`, `Moderate`,
`Low`, `Decay`, `Dried`, `Disease`. These are remapped at load time via
`config.CLASS_TARGET_MAP` into the taxonomy the model actually predicts —
category (`Healthy`/`Moderate`/`Low`) + condition
(`None`/`Dried`/`Decayed`/`Diseased`) + a heuristic 0–10 health-score anchor.
Labeling standards for the raw classes are in
[`../docs/DATASET_LABELING_GUIDE.md`](../docs/DATASET_LABELING_GUIDE.md).

**Is the confidence real?** Yes — `confidence` is a genuine softmax
probability from the trained model, not fabricated. But an uncalibrated
softmax number can still be overconfident. `python -m src.calibrate` measures
this (Expected Calibration Error, before/after reliability diagrams) and
produces `checkpoints/calibration.json`; once that exists, the API also
returns `confidence_calibrated` (temperature-scaled). Re-run it after every
retrain.

**Multiple species:** dataset folders are species-scoped
(`dataset/<species_slug>/...`) and `config.SPECIES_SLUG`/`SPECIES_DISPLAY_NAME`
are the single place species is defined. Today there is only one species and
no trained species classifier — `species` in the API response is a constant.
Adding a second species means adding a sibling `dataset/<new_slug>/` folder
and, eventually, a real species-identification model
(see the roadmap in `../docs/STEP_BY_STEP.md`).
