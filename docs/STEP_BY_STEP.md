# Mantis Vision — Step-by-Step: Setup to Everything

This is the master walkthrough: from an empty checkout to a trained health
classifier serving predictions into the PWA, plus the roadmap for everything
after that (species ID, disease/predator models, damage segmentation).

Repo layout after this scaffold:

```
MantisVision/
  apps/web/          Next.js + TypeScript + Tailwind PWA
  ml/                 Python + PyTorch training/inference pipeline
  docs/                This guide + the dataset labeling standards
```

---

## 0. Prerequisites

- Node.js 20+ and npm
- Python 3.11+
- Git
- (Later) A Supabase project and a Vercel account

Check versions:

```bash
node --version
python3 --version
```

---

## Milestone 1 — Project Setup ✅ (this scaffold)

Already done in this repo:

- [x] Repository structure (`apps/web`, `ml`, `docs`)
- [x] Python environment defined (`ml/requirements.txt`)
- [x] Dataset split structure (`ml/dataset/{train,validation,test}/`, each
      holding `images/`, `masks/<measurement_key>/`, and an
      `annotations.jsonl` manifest — see Milestone 2; not species-scoped,
      species is a per-image classification column like any other)
- [x] Logging (`ml/src/utils/logger.py`)
- [x] Reproducibility via fixed seeds (`ml/src/utils/seed.py`, `config.py: seed=42`)

To activate it yourself:

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Milestone 2 — Data Pipeline

### 2.1 Read the labeling standards first

Read [`docs/DATASET_LABELING_GUIDE.md`](./DATASET_LABELING_GUIDE.md) before
labeling anything. Consistent labels matter more than volume.

### 2.2 Labeling happens through the admin panel, not raw folders

Per-image ground truth is **column annotations against the active
measurement schema** (admin-editable at `/member/schema`), not folder names —
see `docs/DATASET_LABELING_GUIDE.md` for the schema shape and labeling rules.
In practice you label by uploading photos through `/member/dataset`, which
renders one control per active measurement (dropdown, number input, or mask
upload) and writes the result to Supabase; the "Retrain" button then
materializes everything into the manifest format below and trains on it —
you don't usually touch `ml/dataset/` by hand.

### 2.3 The dataset lives on Kaggle, not in git

Photo datasets grow large fast and don't belong in git history. The
retraining pipeline can additionally layer an archived **Kaggle Dataset** in
underneath the admin-labeled images, synced with `ml/scripts/kaggle_sync.py`.
`.gitignore` already excludes the actual dataset contents under
`ml/dataset/{train,validation,test}/` — only the directory structure
(`.gitkeep`) is committed.

**One-time setup:**

1. Create a Kaggle account, then go to kaggle.com/settings → API → "Create
   New Token" and copy the username/key token shown there.
2. Set them as environment variables:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_key
   ```
   (If your account instead downloads a `kaggle.json` file, place it at
   `~/.kaggle/kaggle.json` with `chmod 600` — either method works.)
3. `pip install -r requirements.txt` (already includes the `kaggle` package).

A split directory that `kaggle_sync.py` pushes/pulls looks like:

```
ml/dataset/
  train/       images/*.jpg  masks/<measurement_key>/*.png  annotations.jsonl
  validation/  images/*.jpg  masks/<measurement_key>/*.png  annotations.jsonl
  test/        images/*.jpg  masks/<measurement_key>/*.png  annotations.jsonl
```

`ml/scripts/split_dataset.py` exposes the generic fixed-seed 70/15/15
splitter (`split_class`) that the retrain orchestration
(`retrain_and_report.py`) uses when assembling a fresh split from Supabase
rows — it's a library function now, not a standalone CLI for importing raw
folders.

**Push the dataset to Kaggle** (first time creates it, every time after that
pushes a new version — Kaggle keeps the version history for you):

```bash
# first time only
cp ml/dataset/dataset-metadata.json.example ml/dataset/dataset-metadata.json
# edit "id" in that file to <your-kaggle-username>/mantis-vision-kappaphycus-health
git add ml/dataset/dataset-metadata.json && git commit -m "Add Kaggle dataset id"
python scripts/kaggle_sync.py init

# every time after adding more labeled photos
python scripts/kaggle_sync.py push -m "Add 50 more Disease examples"
```

**Pull the dataset onto a fresh machine / training environment** (no need to
re-label anything, or to commit images to git at all):

```bash
python scripts/kaggle_sync.py download --dataset <your-kaggle-username>/mantis-vision-kappaphycus-health
```

**Training in Google Colab instead of a local machine?** Use
[`docs/colab/MantisVision_Training.ipynb`](./colab/MantisVision_Training.ipynb)
— open it at colab.research.google.com (File → Upload notebook, or open
directly from GitHub), set the runtime to GPU, and run the cells top to
bottom. It clones this repo, installs dependencies, lets you upload a zip of
your `train/validation/test` folders directly through a browser file picker
(no Kaggle account needed), trains, evaluates, and lets you download the
resulting checkpoint before the Colab session ends (Colab runtimes are
ephemeral — nothing persists after you disconnect unless you download it,
which means you'll need to re-upload the dataset zip each new session).

Kaggle (`scripts/kaggle_sync.py`) is still there as an alternative once the
dataset grows large enough that re-uploading a zip every Colab session gets
tedious — it's not required for the notebook above.

### 2.4 Validate the dataset

```bash
python -m src.data.validate_dataset
```

This checks every image referenced in each split's `annotations.jsonl`
manifest opens correctly (catches corrupt files before they crash training),
confirms the primary classification's background class is represented in
every split, and flags classes that are severely underrepresented relative
to the largest class.

### 2.5 What the pipeline does under the hood

`src/data/transforms.py` + `src/data/dataset.py` implement:

```
Images -> Resize -> Normalize (ImageNet stats) -> Augmentation (train only) -> DataLoader
```

Augmentations used (all biologically safe — no vertical flips or hue-inverting
transforms that would change what a symptom looks like): random crop,
horizontal flip, rotation, brightness/contrast jitter, Gaussian noise.

---

## Milestone 3 — Baseline Model

Already implemented in `ml/src/models/efficientnet.py`:

- Transfer learning from `EfficientNet-B0` pretrained on ImageNet (per spec:
  best accuracy/size tradeoff for the first model — not trained from
  scratch), a single shared backbone for every measurement.
- `build_model(schema, freeze_backbone=True)` freezes the pretrained backbone
  and grows one fresh head per measurement in the active `Schema`:
  classification (n-way logits), regression (a sigmoid-squashed scalar), or
  segmentation (a lightweight upsampling decoder off the backbone's feature
  map). Adding a measurement in the schema editor means a new head appears
  automatically next training run — no change to this file.
- `unfreeze_backbone()` is called after the heads have stabilized, to
  fine-tune the whole network.

To try a different backbone later (`EfficientNetV2-S`, `ConvNeXt-Tiny`,
`MobileNetV3`), that's the only file to change — the training loop doesn't
care which architecture it's driving.

---

## Milestone 4 — Training & Evaluation

### 4.1 Train

```bash
cd ml
python -m src.train
```

This runs the two-phase schedule from `config.py`:

1. **Frozen phase** (`frozen_epochs`, default 10): only the classifier head
   trains.
2. **Fine-tune phase** (`finetune_epochs`, default 20): backbone unfrozen,
   whole network trains at a lower learning rate.

Early stopping (`early_stopping_patience`, default 6 epochs with no
improvement) applies independently within each phase, per measurement. The
best checkpoint (lowest validation loss) is saved to
`ml/checkpoints/best_model.pt` with the **schema it was trained against
baked in** — so a checkpoint is always decoded using its own saved schema's
class order and thresholds, not whatever schema happens to be active later.
This is also what makes hot-swapping a promoted checkpoint safe: promoting a
run atomically swaps both the weights and the head/class configuration.

Logs go to `ml/logs/train.log` as well as the console.

### 4.2 Evaluate

```bash
python -m src.evaluate
```

Produces, per measurement, per the spec's "don't rely only on accuracy"
requirement:

- **Classification** (primary classification only, for the confusion
  matrix): accuracy, macro/weighted precision, recall, F1, per-class
  breakdown, confusion matrix image → `ml/reports/confusion_matrix.png`,
  one-vs-rest ROC AUC per class (when the test split has enough samples)
- **Regression**: mean absolute error on the samples where a ground-truth
  value exists
- **Segmentation**: mean IoU per mask class, on samples with a ground-truth
  mask
- Full results JSON → `ml/reports/evaluation_results.json`

Use the per-class breakdown to see which classes need more data — e.g. if
`Disease` sits at 81% while `Dried` is at 99%, that's a signal to collect more
`Disease` photos, not to tune hyperparameters further.

---

## Milestone 5 — Explainability

```bash
python -m src.gradcam path/to/photo.jpg
```

Saves `photo.gradcam.png` next to the input — a heatmap over the regions
that most influenced the prediction. Grad-CAM is also wired into the
inference API (Milestone 6) so every prediction returned to the app includes
its heatmap, not just a bare label.

Inspect both correct and incorrect predictions on the test set (not just
random samples) — misclassifications with a highlighted region make it
obvious whether the model is looking at the actual symptom or a background
artifact.

---

## Milestone 6 — Inference API

```bash
cd ml
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

- `GET /health` → `{status, model_loaded, species_classes, measurements}` —
  `species_classes` is the full list of species the loaded schema's "species"
  measurement knows about (not a single "active" one); `measurements` is the
  list of measurement keys the currently-loaded checkpoint's schema defines
- `POST /predict` → multipart form with a `file` field, returns a legacy flat
  shape (kept stable for the current PWA) plus a generic, forward-looking
  `measurements` map — one entry per schema measurement:

```json
{
  "species": "Kappaphycus alvarezii",
  "is_seaweed": true,
  "condition": "Disease",
  "health": "Moderate",
  "health_score": 62.1,
  "confidence": 0.974,
  "disease_subtype": "IceIce",
  "dried_pct": null,
  "decayed_pct": null,
  "explanation": "Minor bleaching on branches and early tissue degradation observed.",
  "recommendation": "Increase water movement. Inspect for grazers and early disease signs.",
  "gradcam_png_base64": "...",
  "measurements": {
    "condition": {"type": "classification", "value": "Disease", "confidence": 0.974, "explanation": "...", "recommendation": "...", "coverage": null, "mask_png_base64": null},
    "disease_subtype": {"type": "classification", "value": "IceIce", "confidence": 0.88, "explanation": null, "recommendation": null, "coverage": null, "mask_png_base64": null},
    "health_score": {"type": "regression", "value": 62.1, "confidence": null, "explanation": null, "recommendation": null, "coverage": null, "mask_png_base64": null}
  }
}
```

- `POST /admin/reload` → hot-swaps the running checkpoint (weights + schema)
  from a `model_url` without restarting the process; requires a
  `RELOAD_TOKEN` bearer token. Called by the admin "Promote" action — see
  [`DEPLOY_ML_API.md`](DEPLOY_ML_API.md).

Test `/predict` directly:

```bash
curl -F "file=@/path/to/photo.jpg" http://localhost:8000/predict
```

`species` in the response is a real predicted classification (the schema's
"species" measurement), not a fixed constant — the admin adds one class per
species they collect, extensible like `disease`, with no "active species"
concept.

---

## Milestone 7 — Application Integration

### 7.1 Run the web app against the local API

```bash
cd apps/web
cp .env.example .env.local     # set ML_API_URL=http://localhost:8000
npm install
npm run dev
```

Open `http://localhost:3000`, upload a photo, and confirm the result card
shows species/health/confidence/explanation/recommendation plus the Grad-CAM
overlay. The app's `/api/predict` route (`src/app/api/predict/route.ts`)
proxies the upload to the Python service — the browser never talks to the ML
API directly.

Already verified working in this scaffold: `npm install`, `npm run
typecheck`, and `npm run build` all pass cleanly.

### 7.2 Add real PWA icons

Replace the placeholders referenced in `apps/web/public/manifest.webmanifest`
— drop `icon-192.png` and `icon-512.png` into `apps/web/public/icons/`.

### 7.3 Set up Supabase

1. Create a project at supabase.com.
2. Create a storage bucket (e.g. `specimen-photos`) for uploaded images.
3. Create a table for prediction history, e.g.:

   ```sql
   create table predictions (
     id uuid primary key default gen_random_uuid(),
     created_at timestamptz default now(),
     species text not null,
     health text not null,
     confidence numeric not null,
     explanation text,
     recommendation text,
     image_path text,
     gps point,
     farm text,
     water_temperature_c numeric,
     salinity_ppt numeric
   );
   ```

4. Copy the project URL and anon key into `apps/web/.env.local`
   (`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`) and the
   service role key server-side only (`SUPABASE_SERVICE_ROLE_KEY`) if you add
   server-side writes.
5. Wire `apps/web/src/app/api/predict/route.ts` to also persist each
   prediction + uploaded image (not implemented yet in this scaffold — the
   `@supabase/supabase-js` dependency is already in `package.json`).

### 7.4 Export the model for portability

```bash
cd ml
python scripts/export_model.py
```

Produces `ml/checkpoints/health_classifier.onnx` +
`ml/checkpoints/class_names.json`. From ONNX you can convert further to
TensorFlow Lite or CoreML for native mobile builds later.

### 7.5 Deploy

- **Web app (`apps/web`)** → Vercel. `vercel --cwd apps/web` or connect the
  repo in the Vercel dashboard with the root directory set to `apps/web`.
  Set `ML_API_URL` and the Supabase env vars as Vercel project env vars.
  Also set `NEXT_PUBLIC_SITE_URL` (e.g. `https://mantis-vision.vercel.app`)
  — without it, admin team invite links fall back to the request origin and
  can end up pointing at `localhost:3000` if an invite is sent from a local
  dev server. This should also be added to Supabase's Authentication → URL
  Configuration → Redirect URLs (and set as the Site URL there) — if it's
  missing from that allow-list, Supabase ignores the app's `redirectTo` and
  sends invitees to the Site URL instead. As a safety net for exactly that
  misconfiguration, `AuthCallbackRedirect` (mounted in the root layout)
  detects an auth token/error landing in the hash of any page and forwards it
  to `/member/password-reset`, so invite links still work even if the
  Supabase-side allow-list wasn't updated.
- **ML inference API (`ml`)** → Vercel's Node/edge runtime is not a fit for a
  PyTorch service. Deploy `ml/src/api` as its own service instead — a small
  container on Fly.io/Render/Railway, or a GPU-less CPU box is fine for
  EfficientNet-B0 at this scale. Point the web app's `ML_API_URL` at it. See
  [`docs/DEPLOY_ML_API.md`](DEPLOY_ML_API.md) for the exact Render setup
  (`ml/Dockerfile` is already in the repo).

---

## End-to-end local checklist

- [ ] `ml`: venv created, `pip install -r requirements.txt` succeeds
- [ ] Kaggle credentials configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`)
- [ ] Dataset pushed to Kaggle (`kaggle_sync.py init`/`push`) and/or pulled down (`kaggle_sync.py download`)
- [ ] `python -m src.data.validate_dataset` reports no missing/corrupt files
- [ ] `python -m src.train` completes, `ml/checkpoints/best_model.pt` exists
- [ ] `python -m src.evaluate` produces `ml/reports/evaluation_results.json` and a confusion matrix
- [ ] `uvicorn src.api.main:app --port 8000` serves `/health` and `/predict`
- [ ] `apps/web`: `npm install && npm run dev` runs, uploads a photo, and renders a result

---

## Long-Term Roadmap (Future Expansion)

Unlike the original folder-taxonomy design (where each new capability meant a
new model, a new module, and a new API endpoint), the schema-driven
architecture means most of the roadmap below is **an admin-UI schema edit —
add a measurement, relabel some photos, retrain** — not new code:

```
Image
  |
  v
condition (classification, incl. Background/no-subject)
  |
  v
health_score (regression)  ->  display level (Healthy/Moderate/Low)
  |
If Disease:
  disease_subtype (classification, applies_when condition == "Disease")
```

Every measurement below can be added the same way `disease_subtype` and
`health_score` already are — as an entry in the measurement schema — with no
change to the model, dataset loader, losses, or predictor:

1. **Species Identification** — done: `species` is a classification
   measurement in the default schema, extensible from `/member/schema` (add a
   class per species with labeled data, e.g. `Eucheuma_denticulatum`,
   `Gracilaria`, `Ulva`, `Sargassum`) — no code change needed.
2. **Predator detection** — a new classification measurement (e.g. keyed
   `predator`, `applies_when condition == "Disease"` or its own trigger),
   outputs e.g. `Rabbitfish`, `Sea Urchin`, `Parrotfish`, `Crab`, `Unknown`.
3. **Damage/Decay extent** — already modeled today as the `dried_extent`/
   `decayed_extent`-style regression measurements once real percentages are
   collected (currently masked/untrained — no data yet).
4. **Region/pixel-level detection** — a segmentation measurement (like the
   default schema's `biofouling` slot), outputs per-pixel class coverage +
   an overlay mask; requires mask ground truth (upload via `/member/dataset`).
5. **Treatment Recommendation** — already schema-driven: preset
   `explanation`/`recommendation` copy lives per class, editable in
   `/member/schema`, and travels with the checkpoint on promotion (no
   separate lookup file to maintain).
6. **Farm Analytics** — aggregate prediction history (once an end-user
   prediction-history table is wired up in Supabase, see 7.3 above) across
   regions/time for trend detection.

The one thing schema edits can't do alone is a genuinely new *input
modality* (e.g. a second photo angle, sensor readings) — that still needs
its own dataset/model plumbing.
