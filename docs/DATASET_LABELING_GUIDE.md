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
 "measurements": {"condition": "Healthy", "health_score": 82.4},
 "masks": {"biofouling": "0001.png"}}
```

Each key in `measurements` is a measurement's `key` (auto-derived from its
label in the admin editor — see below); the value is a class name string
(classification) or a number (regression). `masks` maps a segmentation
measurement's key to a mask filename under `masks/<key>/`. **Every
measurement is optional per image** — one missing from a row simply
contributes nothing to that head's loss for that image. This is how "no lab
data for this measurement yet" and conditional measurements (e.g.
`disease_subtype` only meaningful when `condition == "Disease"`) both work,
for free, with no special-casing.

In practice you label through the admin upload form
(`/admin/dataset`), which renders one control per active measurement
(dropdown for classification, number input for regression, mask upload for
segmentation) and writes the resulting `measurements` map to Supabase; the
retrain pipeline then materializes it into `annotations.jsonl` for training.
Hand-editing the manifest directly is only for local experiments.

## The default measurement schema

Out of the box (`DEFAULT_SCHEMA` in both `ml/config.py` and
`apps/web/src/lib/schema.ts`) there are three measurements:

| Measurement | Type | Values |
|---|---|---|
| `condition` | classification | `Background`, `Healthy`, `Disease`, `Decay`, `Dried` |
| `disease_subtype` | classification, `applies_when condition == "Disease"` | `IceIce`, `Epiphyte`, `Bacterial`, `Bleaching`, `Unknown` |
| `health_score` | regression, 0–100 | a continuous score |

`condition` is the **primary classifier**: its `Background` class is the
schema's designated "no subject" class (set once, schema-wide, in the admin
editor's "Primary classifier" section — not a per-measurement toggle). Fill
it with **diverse non-seaweed images**: empty underwater scenes, rocks,
ropes, sand, other organisms, blurry/dark frames with no specimen. This is
what teaches the model to say "no seaweed detected" instead of hallucinating
a health assessment on an irrelevant photo.

Severity (Moderate vs. Low) is **not a label you assign**. It's a display-only
bucket derived at inference purely from the regressed `health_score` against
two admin-editable thresholds (`health_moderate_min`, `health_healthy_min` —
"Display thresholds" in the schema editor): at or above `health_healthy_min`
→ "Healthy", at or above `health_moderate_min` → "Moderate", otherwise
"Low". This applies uniformly to any non-background condition — there's no
per-condition special case, so label a genuinely healthy-looking specimen
with a high `health_score` and it will show "Healthy" regardless of which
`condition` class it's under.

Adding a new measurement (a lab value like moisture or gel strength, another
segmentation target, a named-disease classifier) is an edit in `/admin/schema`,
not a code change — the model, losses, dataset loader, and predictor all grow
the new head generically from the schema. It stays masked (untrained, no
effect on other heads) until images with real values for it exist.

Swapping the active species is also a schema edit (the Species section);
`active_species_slug` prefixes the dataset directory.

Split ratio: **70% train / 15% validation / 15% test**, applied by the
retrain pipeline (`ml/scripts/split_dataset.py`'s `split_class`) with a fixed
seed for reproducibility. Run `python scripts/build_label_map.py` after
pulling a dataset locally to generate `ml/metadata/label_map.csv`, a
human-auditable "what does each image actually train on" audit trail (per
schema measurement, per image).

## Label by content, not quality

**Label what is actually in the image, never the image quality.** There is no
"blurry" or "low-quality" class. A soft, poorly-lit photo of a healthy
specimen is still `Healthy`. The model learns to tolerate quality variation
through augmentation (blur, brightness/contrast jitter, and sensor noise are
all applied during training — see `ml/src/data/transforms.py`), not through a
separate label. Only reject an image if you genuinely can't tell what it
shows.

## Condition definitions

### Healthy
Bright coloration, no whitening, no broken branches.

### Decay
Tissue melting, brown patches, rot.

### Dried
Largely dried out/bleached, detached from the line, little living tissue.

### Disease
Visible lesions / infection symptoms not explained by grazing or general
decay. Assign a `disease_subtype`.

### Background
No seaweed specimen in frame — the primary classifier's negative class (see
above).

## Addressing label noise

Label noise (especially from ambiguous or low-quality images) is the biggest
threat to model quality. Defenses, in order:

1. **Clear, unambiguous rules.** Use the definitions above the same way every
   time. When two conditions are plausible, discard or flag for a second
   annotator — don't guess.
2. **Two-annotator agreement for ambiguous cases** (e.g. Decay vs. Dried, or
   a borderline `health_score`). Resolve disagreements by discussion, not
   silent averaging.
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
