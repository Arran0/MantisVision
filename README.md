# Mantis Vision

Mantis Vision is a Progressive Web App aiming to become the "Google Lens for
Seaweed": photograph a specimen and get species identification, health
assessment, disease/predator detection, damage estimation, and treatment
recommendations.

**Phase 1 (current scope):** health assessment for one species,
*Kappaphycus alvarezii* — condition (`Healthy`, `Disease`, `Decay`, `Dried`,
plus a `Background`/no-subject class), a `disease_subtype` when diseased, and
a continuous 0–100 `health_score` that a display-level (Healthy/Moderate/Low)
is derived from at inference.

The system is **schema-driven**, not hardcoded: a single admin-editable
**measurement schema** (`/admin/schema`) defines every per-image measurement
the model predicts — classification, regression, or segmentation — and a
shared EfficientNet-B0 backbone grows one head per measurement automatically.
Adding species ID, a disease model, predator detection, or damage
segmentation later is a schema edit (new measurement), not a rebuild — see
[`docs/STEP_BY_STEP.md`](docs/STEP_BY_STEP.md#long-term-roadmap-future-expansion)
for how those plug in.

## Start here

- **New to this repo?** Read [`docs/STEP_BY_STEP.md`](docs/STEP_BY_STEP.md) —
  it walks through everything from environment setup to training, evaluation,
  the inference API, the web app, and deployment.
- **Labeling photos?** Read
  [`docs/DATASET_LABELING_GUIDE.md`](docs/DATASET_LABELING_GUIDE.md) first.
- **Just want the ML commands?** See [`ml/README.md`](ml/README.md).

## Structure

```
apps/web/   Next.js + TypeScript + Tailwind CSS PWA (deployed on Vercel)
ml/          Python + PyTorch training/evaluation/inference pipeline
docs/        Setup guide + dataset labeling standards
```

## Stack

Frontend: TypeScript, Tailwind CSS, PWA · Backend: Node.js (Next.js API
routes) + Python (FastAPI inference service) · Data: Supabase · Deployment:
Vercel · ML: Python, PyTorch, EfficientNet-B0 transfer learning, Grad-CAM.
