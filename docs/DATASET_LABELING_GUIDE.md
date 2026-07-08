# Dataset Labeling Guide — Kappaphycus alvarezii Health Classification

The dataset is the most important part of this project. A model is only as
good as the consistency of its labels — apply these rules the same way every
time, and when a photo is ambiguous between two classes, discard it or flag
it for a second annotator rather than guessing.

## Folder structure

```
ml/dataset/
  kappaphycus_alvarezii/                    # <species_slug> — see "Species scoping" below
    train/{Healthy,Moderate,Low,Decay,Dried,Disease}/
    validation/{Healthy,Moderate,Low,Decay,Dried,Disease}/
    test/{Healthy,Moderate,Low,Decay,Dried,Disease}/
```

Phase 1 currently trains on 6 raw classes — `Predator` (grazed) was dropped
for now since there's no real photo data for it yet. It can be added back
later as a 7th class the moment labeled examples exist; nothing else in the
pipeline needs to change to support that (`config.py`'s `CLASS_NAMES` is the
only place the class list is defined).

### Species scoping

The dataset directory is nested one level under a species slug
(`dataset/<species_slug>/...`) so a future second species can be added as a
sibling folder without restructuring anything. Today only
`kappaphycus_alvarezii` exists (`config.SPECIES_SLUG`) — this is a
placeholder, not a trained species classifier; the app currently identifies
only this one species. See the roadmap in `docs/STEP_BY_STEP.md` for when a
real species-identification model gets added.

### From raw class to category/condition/score

Labelers still only ever assign one of the 6 raw class names above to a
photo — nothing changes about the labeling process itself. Internally,
`config.CLASS_TARGET_MAP` remaps each raw class into what the model actually
predicts: a top-level **category** (`Healthy`/`Moderate`/`Low`), an optional
**condition** tag (`Dried`/`Decayed`/`Diseased`, only set when category !=
Healthy), and a heuristic **health-score anchor** in `[0, 10]` used to train
the score-regression head:

| Raw folder | Category | Condition | Score anchor |
|---|---|---|---|
| Healthy | Healthy | — | 9.0 |
| Moderate | Moderate | — | 6.0 |
| Low | Low | — | 3.0 |
| Decay | Low | Decayed | 2.0 |
| Disease | Low | Diseased | 1.5 |
| Dried | Low | Dried | 0.5 |

`Decay`, `Dried`, and `Disease` all map to category **Low** — the
descriptions below (rot, zero living tissue, active lesions) all describe
severe/advanced symptoms consistent with the Low tier, and no photos
currently distinguish a moderate-severity vs. low-severity case of any of the
three. The score anchors are a documented heuristic derived from this guide's
own severity language, not measured biological ground truth — see
`config.py`'s comments for the full justification, and treat them as a first
draft to revise once real expert-scored data exists.

Folder names must match exactly (case-sensitive) — they become the model's
class labels via `torchvision.datasets.ImageFolder`.

Split ratio: **70% train / 15% validation / 15% test**. Use
`ml/scripts/split_dataset.py` to split a flat labeled folder automatically
with a fixed seed (reproducible, so re-running it doesn't reshuffle your
splits).

## Minimum metadata

Every image needs at minimum:

| filename | label |
|---|---|
| healthy_001.jpg | Healthy |
| dried_001.jpg | Dried |

Use `ml/metadata/labels_template.csv` as the starting spreadsheet. Add these
columns as they become available — they aren't required for Phase 1 training
but unlock per-farm/per-region analysis later: `species`, `gps`, `farm`,
`date`, `camera`, `water_temperature_c`, `salinity_ppt`, `depth_m`, `time`,
`annotator`, `notes`.

## Class definitions

### Healthy
- Bright green coloration
- No whitening
- No broken branches

### Moderate
- Slight whitening
- Small tissue loss
- Still actively growing

### Low
- Significant discoloration
- Reduced branching compared to a healthy thallus

### Decay
- Tissue melting
- Brown patches
- Rot

### Dried
- Completely dried out/bleached white
- Detached from the line/substrate
- No living tissue anywhere in frame

### Disease
- Visible lesions
- Infection symptoms (not explained by grazing or general decay)

## Annotation practices

1. **One class per image.** If a specimen shows both decay and disease
   symptoms, pick the dominant/most obvious symptom, or better, exclude it
   from the Phase 1 dataset and flag it for the future multi-label Disease
   model.
2. **Consistent framing.** Fill the frame with the specimen; avoid heavy
   background clutter that could let the model shortcut on non-biological
   cues (e.g. always photographing "Dried" samples on a particular boat
   deck).
3. **Two-annotator agreement for ambiguous classes** (Low vs. Moderate,
   Decay vs. Dried) where possible. Disagreements should be discussed and
   resolved before the image enters the training set, not silently averaged.
4. **Do not relabel to fix class imbalance.** If a class is underrepresented,
   collect more images of that class — don't reclassify borderline images
   into it. `python -m src.data.validate_dataset` (see `ml/README.md`) flags
   underrepresented classes after every dataset update.
5. **Reject corrupt/blurry/out-of-focus images** rather than trying to force
   a label onto unusable data.
