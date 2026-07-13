# Dataset Labeling Guide — Seaweed Multi-Head Model

The dataset is the most important part of this project. A model is only as
good as the consistency of its labels — apply these rules the same way every
time, and when a photo is ambiguous, discard it or flag it for a second
annotator rather than guessing.

## Folder structure

The dataset is organized under a single active species slug, with **flat,
leaf class folders** inside each split (nested subfolders do NOT work with
`torchvision.datasets.ImageFolder`):

```
ml/dataset/<species_slug>/
  train/       <class_folder>/*.jpg
  validation/  <class_folder>/*.jpg
  test/        <class_folder>/*.jpg
```

Class-folder names encode structured labels (parsed in
`ml/src/data/labels.py`). **Severity comes before the condition token**, so
every non-Healthy, non-Background folder has the shape
`<species_slug>_<Severity>_<Decay|Dried|Disease>...`:

| Condition | Folder name |
|---|---|
| Healthy | `<species_slug>_Healthy` |
| Decay | `<species_slug>_Low_Decay` |
| Dried | `<species_slug>_Low_Dried` |
| Disease | `<species_slug>_<Severity>_Disease_<Subtype>[_<DiseaseName>]` |
| Background (no seaweed) | `Background` |

Example: `Kappaphycus_alvarezii_Moderate_Disease_IceIce`,
`Kappaphycus_alvarezii_Low_Disease_Bacterial_Vibrio_sp`, `Background`.

- **Severity** ∈ `Moderate` (small/localized patch) or `Low` (widespread) —
  but **Decay and Dried only ever use `Low`**. There is no
  Moderate/Healthy-severity bucket for them: by definition a decayed or dried
  specimen is already at the bottom of the health range, so only one bucket
  each exists. Disease uses both `Moderate` and `Low`. Severity sets the
  health-score training anchor; at inference the model re-derives Disease's
  Moderate/Low from its regressed 0–100 score (Decay/Dried are always "Low").
- **Subtype** (Disease only) ∈ `IceIce`, `Epiphyte`, `Bacterial`, `Bleaching`,
  `Unknown`.
- **DiseaseName** (Disease only) is a free-form trailing token (e.g. a
  specific pathogen). It's parsed and recorded as **metadata only** — it is
  not its own model head yet.

The `Background` class is the N+1 negative class. Fill it with **diverse
non-seaweed images**: empty underwater scenes, rocks, ropes, sand, other
organisms, blurry/dark frames with no specimen. This is what teaches the
model to say "no seaweed detected" instead of hallucinating a health
assessment on an irrelevant photo.

Swapping the active species is a one-line change (`SPECIES` in
`ml/config.py`); the slug then prefixes every class folder.

Split ratio: **70% train / 15% validation / 15% test**. Use
`ml/scripts/split_dataset.py` to split a flat labeled folder automatically
with a fixed seed. Run `python scripts/build_label_map.py` afterward to
(re)generate `ml/metadata/label_map.csv`, the human-auditable
folder → integer-ID map the model trains on.

## Label by content, not quality

**Label what is actually in the image, never the image quality.** There is no
"blurry" or "low-quality" class. A soft, poorly-lit photo of a healthy
specimen is still `Healthy`. The model learns to tolerate quality variation
through augmentation (blur, brightness/contrast jitter, and sensor noise are
all applied during training — see `ml/src/data/transforms.py`), not through a
separate label. Only reject an image if you genuinely can't tell what it
shows.

## Multiple species / diseases later

When expanding beyond one species or adding named diseases, keep the same
flat-folder + label-map approach: add new class folders following the naming
convention and re-run `build_label_map.py`. The CSV makes the
folder → integer-ID mapping explicit and reviewable rather than implicit in
code.

## Minimum metadata

Every image needs at minimum a filename and a `class_folder` (or the
condition + Disease fields it's built from). Use
`ml/metadata/labels_template.csv` as the starting spreadsheet. The optional
numeric columns (`health_score`, `dried_pct`, `decayed_pct`) let you provide
real ground truth for the regression heads; leave them blank to fall back to
the per-condition heuristic anchors in `ml/config.py`. The remaining columns
(`species`, `gps`, `farm`, `date`, `camera`, `water_temperature_c`,
`salinity_ppt`, `depth_m`, `time`, `annotator`, `notes`) unlock
per-farm/per-region analysis later.

## Condition definitions

### Healthy
Bright coloration, no whitening, no broken branches.

### Decay
Tissue melting, brown patches, rot. Always folder severity `Low` — no
Moderate/Healthy variant.

### Dried
Largely dried out/bleached, detached from the line, little living tissue.
Always folder severity `Low` — no Moderate/Healthy variant.

### Disease
Visible lesions / infection symptoms not explained by grazing or general
decay. Assign a **severity** (Moderate = localized, Low = widespread) and a
**subtype**; add a specific disease name if known.

### Background
No seaweed specimen in frame — the negative class (see above).

## Addressing label noise

Label noise (especially from ambiguous or low-quality images) is the biggest
threat to model quality. Defenses, in order:

1. **Clear, unambiguous rules.** Use the definitions above the same way every
   time. When two conditions are plausible, discard or flag for a second
   annotator — don't guess.
2. **Two-annotator agreement for ambiguous cases** (e.g. Decay vs. Dried,
   Moderate vs. Low). Resolve disagreements by discussion, not silent
   averaging.
3. **Augmentation** simulates real-world quality variation so the model
   doesn't overfit to pristine lab photos (already wired into training).
4. **Label smoothing** on the condition head softens the impact of a wrong
   label (already enabled in `ml/src/train.py`). For heavier noise, the
   condition criterion can be swapped for a Generalized Cross-Entropy /
   symmetric loss in `ml/src/losses.py` — the rest of the pipeline is
   unaffected.
5. **Don't relabel to fix class imbalance.** If a class is underrepresented,
   collect more of it. `python -m src.data.validate_dataset` reports
   per-condition counts and flags underrepresented conditions.
