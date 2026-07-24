# Mantis Vision

Mantis Vision is a Progressive Web App aiming to become the **"Google Lens for
Seaweed"**: photograph a specimen and get species identification, health
assessment, disease/predator detection, damage estimation, and treatment
recommendations — with a Grad-CAM heatmap showing what the model looked at.

**Current focus:** health assessment for *Kappaphycus alvarezii*. But the
system is **schema-driven, not hardcoded** — a single admin-editable
*measurement schema* defines every per-image measurement the model predicts,
and the model grows one head per measurement automatically. Adding species ID,
a disease model, predator detection, or damage segmentation is a schema edit,
not a rebuild.

| Doc | What's in it |
|---|---|
| **[INSTALL.md](INSTALL.md)** | Prerequisites, setup, environment variables, running everything locally |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How the pieces fit together, request/data flow, the schema-driven design |
| **[MODEL.md](MODEL.md)** | The ML pipeline: model, training, evaluation, explainability, export |
| **[API.md](API.md)** | Every HTTP endpoint (ML inference service + web API routes) |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Dev workflow, tests, code conventions, how to extend the schema |
| [docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md) | The long-form, milestone-by-milestone walkthrough |
| [docs/DATASET_LABELING_GUIDE.md](docs/DATASET_LABELING_GUIDE.md) | Labeling standards — read before labeling anything |

---

## Overview

A user opens the PWA, uploads a photo of a seaweed specimen, and gets back a
structured assessment: is there seaweed in frame, what species, how healthy,
any disease and its severity, plus preset explanation/recommendation copy and
a Grad-CAM overlay. Admins manage the model and its data from an authenticated
`/member` dashboard — edit the measurement schema, label photos, trigger a
retrain on GitHub Actions, review metrics, and promote a new checkpoint into
live serving with zero downtime.

There are three cooperating pieces:

- **`apps/web`** — a Next.js PWA (the public analyzer + the admin dashboard).
  Its API routes proxy uploads to the ML service and talk to Supabase.
- **`ml`** — a Python/PyTorch pipeline (data, training, evaluation, Grad-CAM)
  and a FastAPI service that serves predictions.
- **`supabase`** — Postgres schema (migrations) for auth/roles, the
  measurement schema, labeled training data, and model-run history.

## Features

- **Photo → structured assessment.** Species, seaweed presence, health status,
  disease + severity, colour, and a block of lab/quality regressions.
- **Explainability built in.** Every prediction ships a Grad-CAM heatmap, not
  just a label.
- **Schema-driven model.** One shared EfficientNet-B0 backbone grows one head
  per measurement in the active schema — classification, regression, or
  segmentation. New measurements are an admin-UI edit, not code.
- **Admin dashboard** (`/member`, role-gated): edit the measurement schema,
  upload & label photos, manage teammates, retrain, and promote checkpoints.
- **Zero-downtime model promotion.** Promoting a completed retrain hot-swaps
  the live checkpoint (weights + schema) with no restart; a bad checkpoint
  never reaches production.
- **CI retraining.** The "Retrain" button dispatches a GitHub Actions workflow
  so training never competes with live inference traffic.
- **Installable PWA** with offline shell, service worker, and app icons.

## Folder structure

```
MantisVision/
├── apps/web/                 Next.js + TypeScript + Tailwind PWA (deploys to Vercel)
│   ├── src/app/              App Router pages + API routes (/api/predict, /api/member/*)
│   ├── src/components/       UI components (upload, result card, admin panels)
│   ├── src/lib/              Supabase clients, roles, the shared measurement schema
│   └── .env.example          Web environment variables
├── ml/                       Python + PyTorch pipeline and FastAPI inference service
│   ├── config.py             Paths, hyperparameters, and the fallback DEFAULT_SCHEMA
│   ├── src/                  data / models / losses / train / evaluate / gradcam / inference / api
│   ├── scripts/              schema export, dataset sync, retrain orchestration, ONNX export
│   ├── tests/                pytest suite
│   └── requirements*.txt     Full (train) vs. serve-only dependency sets
├── supabase/migrations/      Postgres schema: auth/roles, measurement schema, data, model runs
├── docs/                     Long-form guides (setup, labeling, deployment)
└── README / INSTALL / ARCHITECTURE / MODEL / API / CONTRIBUTING
```

## Technologies used

| Layer | Stack |
|---|---|
| Frontend | TypeScript, Next.js 14 (App Router), React 18, Tailwind CSS, Framer Motion, PWA (service worker + manifest) |
| Web backend | Next.js API routes (Node runtime) |
| ML service | Python, FastAPI, Uvicorn, PyTorch, torchvision |
| Model | EfficientNet-B0 transfer learning, multi-head, Grad-CAM explainability, ONNX export |
| Data / auth | Supabase (Postgres, Auth, Storage, RLS) |
| Dataset storage | Kaggle Datasets (versioned; images stay out of git) |
| CI / automation | GitHub Actions (retrain workflow, Hugging Face Space deploy) |
| Hosting | Vercel (web) · Hugging Face Spaces or Render (ML service) |

## Screenshots

> Screenshots are not committed to the repo yet. Drop images into
> `docs/screenshots/` and replace the placeholders below before handoff — the
> key screens to capture are listed so a reviewer knows what each should show.

| Screen | What it shows |
|---|---|
| Analyzer (home) | Upload/capture a photo and view the result card with Grad-CAM overlay |
| Result card | Species, health, confidence, explanation/recommendation, heatmap |
| Admin · Structure | The measurement-schema editor (add/edit measurements & classes) |
| Admin · Dataset | Upload + label photos against the active schema |
| Admin · Retrain | Trigger a retrain, watch progress, review metrics, promote a run |

## Installation

Full instructions — including every environment variable — are in
**[INSTALL.md](INSTALL.md)**. The short version:

```bash
git clone <your-repo-url> MantisVision
cd MantisVision

# ML service
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web app
cd ../apps/web
cp .env.example .env.local        # then fill in the values (see INSTALL.md)
npm install
```

## Running locally

```bash
# Terminal 1 — ML inference service (needs a trained ml/checkpoints/best_model.pt)
cd ml && source .venv/bin/activate
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — web app
cd apps/web
npm run dev            # http://localhost:3000
```

With `ML_API_URL=http://localhost:8000` in `apps/web/.env.local`, uploading a
photo in the browser proxies through `/api/predict` to the FastAPI service and
renders the result. Training/evaluation commands are in
[MODEL.md](MODEL.md); the admin dashboard needs Supabase configured (see
[INSTALL.md](INSTALL.md)).

## Deployment

- **Web app → Vercel.** Root directory `apps/web`. Set `ML_API_URL`, the
  Supabase vars, `NEXT_PUBLIC_SITE_URL`, and (for retrain) `GITHUB_TOKEN` /
  `GITHUB_REPO`.
- **ML service → Hugging Face Spaces (recommended) or Render.** Both build the
  same [`ml/Dockerfile`](ml/Dockerfile) — it also runs anywhere Docker does
  (see [INSTALL.md → Docker](INSTALL.md#docker-ml-inference-service)). Deploy
  guides: [docs/DEPLOY_HUGGINGFACE.md](docs/DEPLOY_HUGGINGFACE.md) and
  [docs/DEPLOY_ML_API.md](docs/DEPLOY_ML_API.md).
- **Database → Supabase.** Apply everything in `supabase/migrations/`.

A full architecture-level view of these moving parts is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Future roadmap

Because the model is schema-driven, most of the roadmap is an **admin-UI schema
edit — add a measurement, relabel some photos, retrain** — not new code:

1. **Species identification** — already a classification measurement; add a
   class per species you collect.
2. **Predator detection** — a new classification measurement (e.g. gated on a
   parent condition).
3. **Damage / decay extent** — regression measurements once real percentages
   are collected.
4. **Region/pixel-level detection** — a segmentation measurement (needs mask
   ground truth).
5. **Treatment recommendations** — already schema-driven preset copy per class.
6. **Farm analytics** — aggregate end-user prediction history over region/time
   for trend detection (needs a prediction-history table).

The one thing a schema edit can't do alone is a genuinely new *input modality*
(a second photo angle, sensor readings) — that still needs its own data/model
plumbing. See
[docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md#long-term-roadmap-future-expansion)
for the detail.

## License

No license file is committed yet. Add one that reflects how NIOT intends the
project to be used before any public distribution.
