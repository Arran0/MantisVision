# Dataset Labeling Guide — Seaweed Multi-Head Model

The dataset is the most important part of this project. A model is only as
good as the consistency of its labels — apply these rules the same way every
time, and when a photo is ambiguous, discard it or flag it for a second
annotator rather than guessing.

## Column annotations, not class folders

There is no folder-name taxonomy anymore (no
`<slug>_<Severity>_<Condition>_<Subtype>` folders to parse). Every per-image
label is a **column value** against the active **measurement schema**
(admin-editable at `/admin/schema`, or `ml/config.py`'s `DEFAULT_SCHEMA` when
no schema has been saved yet). A split directory looks like:

```
ml/dataset/<species_slug>/
  train/       images/*.jpg  masks/<measurement_key>/*.png  annotations.jsonl
  validation/  images/*.jpg  masks/<measurement_key>/*.png  annotations.jsonl
  test/        images/*.jpg  masks/<measurement_key>/*.png  annotations.jsonl
```

`annotations.jsonl` holds one JSON object per image:

```json
{"filename": "0001.jpg",
 "measurements": {"seaweed_presence": "Yes", "health_status": "Healthy", "dried": 5.0},
 "masks": {"biofouling": "0001.png"}}
```

Each key in `measurements` is a measurement's `key` (auto-derived from its
label in the admin editor — see below); the value is a class name string
(classification) or a number (regression). `masks` maps a segmentation
measurement's key to a mask filename under `masks/<key>/`. **Every
measurement is optional per image** — one missing from a row simply
contributes nothing to that head's loss for that image. This is how "no lab
data for this measurement yet" and conditional measurements (e.g.
`health_status` only meaningful when `seaweed_presence == "Yes"`, or
`disease_severity` only when `disease != "NoDisease"`) both work, for free,
with no special-casing.

In practice you label through the admin upload form
(`/admin/dataset`), which renders one control per active measurement
(dropdown for classification, number input for regression, mask upload for
segmentation) and writes the resulting `measurements` map to Supabase; the
retrain pipeline then materializes it into `annotations.jsonl` for training.
Hand-editing the manifest directly is only for local experiments.

## The default measurement schema

Out of the box (`DEFAULT_SCHEMA` in both `ml/config.py` and
`apps/web/src/lib/schema.ts`) the schema is a fixed, **required** backbone of
columns. The core ones:

| Measurement | Type | Values |
|---|---|---|
| `seaweed_presence` | classification (primary) | `Yes`, `No` |
| `health_status` | classification, `applies_when seaweed_presence == "Yes"` | `Healthy`, `Moderate`, `Low` |
| `disease` | classification, `applies_when seaweed_presence == "Yes"` | `NoDisease`, `IceIce`, `Epiphyte`, `Bacterial`, `Bleaching`, … |
| `disease_severity` | regression 0–100, `applies_when disease != "NoDisease"` | a continuous score |
| `dried` / `decayed` | regression 0–100 | continuous extents |
| `colour` | classification, fixed palette | `Green`, `Red`, `Brown`, `Yellow`, `Orange`, `White`, `Black` |

Plus a block of required lab/quality regressions: `carrageenan_yield` (%),
`gel_strength` (g/cm²), `viscosity` (cP), `daily_growth_rate` (%/day),
per-mineral content `mineral_ca`/`mineral_mg`/`mineral_k`/`mineral_na`
(mg/kg), `caw` (%), `impurities` (%), `sulfate_content` (%),
`acid_insoluble_ash` (%), and `ash_content` (%).

Every one of these is **locked** — the admin Structure editor renders it
read-only and the schema API rejects removing or retyping it — so the dataset
always collects the same required columns. The one exception is `disease`,
whose class list is **extensible**: add one class per named disease you want
the model to recognise (keep the `NoDisease` class as the negative).

`seaweed_presence` is the **primary classifier**: its `No` class (formerly
"Background") is the schema's designated "no seaweed" class. Fill it with
**diverse non-seaweed images**: empty underwater scenes, rocks, ropes, sand,
other organisms, blurry/dark frames with no specimen. This is what teaches
the model to say "no seaweed detected" instead of hallucinating a health
assessment on an irrelevant photo.

`health_status` is now an **explicit label you assign** (`Healthy` /
`Moderate` / `Low`), not a bucket derived from a numeric score. Pick the class
that matches what the specimen actually looks like.

You can still **add your own extra measurements** (another segmentation
target, an additional lab value) in `/admin/schema` — that's an edit, not a
code change; the model, losses, dataset loader, and predictor all grow the new
head generically from the schema. A new measurement stays masked (untrained,
no effect on other heads) until images with real values for it exist.

Species is managed in the schema's **Species** section — add one row per
species you collect and pick it from a dropdown when labeling. There's no
longer an "active species" toggle; `active_species_slug` is kept only as the
dataset-directory prefix and just tracks the first species.

Split ratio: **70% train / 15% validation / 15% test**, applied by the
retrain pipeline (`ml/scripts/split_dataset.py`'s `split_class`) with a fixed
seed for reproducibility. Run `python scripts/build_label_map.py` after
pulling a dataset locally to generate `ml/metadata/label_map.csv`, a
human-auditable "what does each image actually train on" audit trail (per
schema measurement, per image).

## Label by content, not quality

**Label what is actually in the image, never the image quality.** There is no
"blurry" or "low-quality" class. A soft, poorly-lit photo of a healthy
specimen is still `health_status: Healthy`. The model learns to tolerate quality variation
through augmentation (blur, brightness/contrast jitter, and sensor noise are
all applied during training — see `ml/src/data/transforms.py`), not through a
separate label. Only reject an image if you genuinely can't tell what it
shows.

## Column definitions

### `seaweed_presence`
`Yes` when a specimen fills the frame; `No` (no seaweed) for the negative
class — no seaweed specimen in frame.

### `health_status`
- **Healthy** — bright, even coloration, no whitening, no broken branches.
- **Moderate** — some discoloration or minor structural loss, largely intact.
- **Low** — extensive discoloration, tissue loss, or structural breakdown.

### `dried` / `decayed`
0–100 extent scores. `dried`: how bleached/desiccated/detached the tissue is.
`decayed`: how much tissue melting, brown rot, or mushy breakdown is present.

### `disease` / `disease_severity`
`disease` is the visible disease (or `NoDisease` for none) — lesions /
infection symptoms not explained by grazing or general decay. When a disease
is assigned, also give `disease_severity` a 0–100 score.

### `colour`
The dominant observed colour from the fixed palette (green, red, brown,
yellow, orange, white, black).

## Addressing label noise

Label noise (especially from ambiguous or low-quality images) is the biggest
threat to model quality. Defenses, in order:

1. **Clear, unambiguous rules.** Use the definitions above the same way every
   time. When two conditions are plausible, discard or flag for a second
   annotator — don't guess.
2. **Two-annotator agreement for ambiguous cases** (e.g. a high `decayed` vs.
   high `dried` score, or a borderline `health_status`). Resolve disagreements
   by discussion, not silent averaging.
3. **Augmentation** simulates real-world quality variation so the model
   doesn't overfit to pristine lab photos (already wired into training).
4. **Label smoothing** on classification heads softens the impact of a wrong
   label (already enabled in `ml/src/train.py`). For heavier noise, a
   criterion can be swapped for a Generalized Cross-Entropy / symmetric loss
   in `ml/src/losses.py` — the rest of the pipeline is unaffected.
5. **Don't relabel to fix class imbalance.** If a class is underrepresented,
   collect more of it. `python -m src.data.validate_dataset` reports
   per-measurement counts and flags underrepresented classes on the primary
   classification.
