# Mantis Vision

Mantis Vision is a Progressive Web App aiming to become the "Google Lens for
Seaweed": photograph a specimen and get species identification, health
assessment, disease/predator detection, damage estimation, and treatment
recommendations.

**Phase 1 (current scope):** health classification for one species,
*Kappaphycus alvarezii*, into 6 classes — `Healthy`, `Moderate`, `Low`,
`Decay`, `Dried`, `Disease`. (`Predator` was dropped for now, pending real
grazing photos — see `docs/DATASET_LABELING_GUIDE.md`.)

The system is built as independent, composable models rather than one large
model — see [`docs/STEP_BY_STEP.md`](docs/STEP_BY_STEP.md#long-term-roadmap-future-expansion)
for how species ID, disease, predator, and damage-segmentation models plug in
later without changing what's already built.

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
