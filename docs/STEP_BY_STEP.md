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
- [x] Dataset folder structure (`ml/dataset/{train,validation,test}/<7 classes>`)
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

### 2.2 The dataset lives on Kaggle, not in git

Photo datasets grow large fast and don't belong in git history. Instead, the
labeled dataset is a **Kaggle Dataset**, synced with
`ml/scripts/kaggle_sync.py`. `.gitignore` already excludes the actual image
files under `ml/dataset/{train,validation,test}/<Class>/` — only the folder
structure (`.gitkeep`) is committed.

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

**Label photos locally, then get them into `ml/dataset/`:**

- **You have a flat folder of labeled photos** (`raw/Healthy/*.jpg`,
  `raw/Dead/*.jpg`, etc.):

  ```bash
  cd ml
  python scripts/split_dataset.py --source /path/to/raw
  ```

  Copies files into `dataset/{train,validation,test}/<Class>/` using a
  fixed-seed 70/15/15 split (reproducible — re-running with the same source
  files and seed gives the same split every time).

- **You're adding photos incrementally.** Drop them directly into
  `ml/dataset/train/<ClassName>/`, `validation/<ClassName>/`, and
  `test/<ClassName>/` yourself, keeping roughly a 70/15/15 ratio per class.

Class folder names (must match exactly): `Healthy`, `Moderate`, `Low`,
`Decay`, `Dead`, `Predator`, `Disease`.

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
bottom. It clones this repo, installs dependencies, pulls the dataset from
Kaggle, trains, evaluates, and lets you download the resulting checkpoint
before the Colab session ends (Colab runtimes are ephemeral — nothing
persists after you disconnect unless you download it).

### 2.3 Validate the dataset

```bash
python -m src.data.validate_dataset
```

This checks every class folder exists, every image opens correctly (catches
corrupt files before they crash training), and flags classes that are
severely underrepresented relative to the largest class.

### 2.4 What the pipeline does under the hood

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
  scratch).
- `build_model(num_classes, freeze_backbone=True)` freezes the pretrained
  backbone and swaps in a fresh classifier head sized for the 7 health
  classes.
- `unfreeze_backbone()` is called after the head has stabilized, to fine-tune
  the whole network.

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
improvement) applies independently within each phase. The best checkpoint
(lowest validation loss) is saved to `ml/checkpoints/best_model.pt`, along
with the exact class-index order the model was trained with — always decode
predictions using that saved order, not `config.CLASS_NAMES`' display order
(see the comment in `src/data/dataset.py` for why).

Logs go to `ml/logs/train.log` as well as the console.

### 4.2 Evaluate

```bash
python -m src.evaluate
```

Produces, per the spec's "don't rely only on accuracy" requirement:

- Accuracy, macro/weighted precision, recall, F1
- Per-class accuracy, precision, recall, F1
- Confusion matrix image → `ml/reports/confusion_matrix.png`
- One-vs-rest ROC AUC per class (when the test split has enough samples)
- Full results JSON → `ml/reports/evaluation_results.json`

Use the per-class breakdown to see which classes need more data — e.g. if
`Disease` sits at 81% while `Dead` is at 99%, that's a signal to collect more
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

- `GET /health` → `{status, model_loaded, classes}`
- `POST /predict` → multipart form with a `file` field, returns:

```json
{
  "species": "Kappaphycus alvarezii",
  "health": "Moderate",
  "confidence": 0.974,
  "explanation": "Minor bleaching on branches and early tissue degradation observed.",
  "recommendation": "Increase water movement. Inspect for grazers and early disease signs.",
  "gradcam_png_base64": "..."
}
```

Test it directly:

```bash
curl -F "file=@/path/to/photo.jpg" http://localhost:8000/predict
```

The response shape already includes `species` so that real species
identification (Milestone 7+/long-term) slots in without a breaking change —
today it's a constant (`SPECIES_NAME` in `src/inference/predictor.py`)
because Phase 1 supports only one species.

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
- **ML inference API (`ml`)** → Vercel's Node/edge runtime is not a fit for a
  PyTorch service. Deploy `ml/src/api` as its own service instead — a small
  container on Fly.io/Render/Railway, or a GPU-less CPU box is fine for
  EfficientNet-B0 at this scale. Point the web app's `ML_API_URL` at it.

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

The architecture is deliberately modular — every block below is its own
independently trainable/deployable model, not a monolith:

```
Image
  |
  v
Species Identification
  |
  +---------------------------+
  |                           |
Kappaphycus alvarezii   Eucheuma denticulatum (etc.)
  |
  v
Health Classification  <-- this scaffold builds exactly this, for one species
  |
  v
If unhealthy:
  +--------+-----------+------------------+
  v        v           v
Disease  Predator   Damage Estimation (segmentation, U-Net, % damaged)
Model     Model
```

Roadmap, in the order the spec recommends tackling them:

1. **Species Identification** — a second classifier run before health
   classification; add as `ml/src/models/species_classifier.py` and a
   `/predict/species` (or a combined pipeline) endpoint. Becomes the first
   stage once more species are supported (`Eucheuma denticulatum`,
   `Gracilaria`, `Ulva`, `Sargassum`).
2. **Disease Model** — runs only when `health != Healthy`. New classifier,
   outputs e.g. `Ice-Ice Disease`, `Epiphyte Infection`, `Bacterial Disease`,
   `Unknown`.
3. **Predator Model** — same trigger condition, outputs e.g. `Rabbitfish`,
   `Sea Urchin`, `Parrotfish`, `Crab`, `Unknown`.
4. **Damage/Decay Estimation** — not a classifier: image → segmentation
   (U-Net) → percentage of tissue damaged (e.g. "Decay: 18.7%").
5. **Region Detection** — another segmentation model, outputs per-pixel
   `Healthy Tissue` / `Damaged Tissue` / `Predator Bite` / `Disease Spot`.
6. **Treatment Recommendation** — currently a static lookup
   (`src/inference/explanations.py`); can later be model- or rule-engine
   driven per disease/predator/region combination.
7. **Farm Analytics** — aggregate prediction history (once Supabase is
   wired up) across farms/regions/time for trend detection.

Each new model should follow the same pattern already established here: its
own module under `ml/src/models/`, its own training/evaluation scripts (or
shared ones parameterized by config), and an additive API endpoint — never a
change to an existing model's contract.
