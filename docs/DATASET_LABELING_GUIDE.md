# Dataset Labeling Guide — Kappaphycus alvarezii Health Classification

The dataset is the most important part of this project. A model is only as
good as the consistency of its labels — apply these rules the same way every
time, and when a photo is ambiguous between two classes, discard it or flag
it for a second annotator rather than guessing.

## Folder structure

```
ml/dataset/
  kappaphycus_alvarezii/                    # <species_slug> — see "Species scoping" below
    train/{Healthy,Moderate,Low,Decay,Dried}/
    train/Disease_<Severity>[_<Subtype>]/    # e.g. Disease_Moderate_IceIce, Disease_Low_Unknown
    validation/{Healthy,Moderate,Low,Decay,Dried}/
    validation/Disease_<Severity>[_<Subtype>]/
    test/{Healthy,Moderate,Low,Decay,Dried}/
    test/Disease_<Severity>[_<Subtype>]/
```

Phase 1 trains on 5 fixed raw classes (`Healthy`, `Moderate`, `Low`, `Decay`,
`Dried`) plus any number of `Disease_<Severity>[_<Subtype>]` folders — see
"Disease folders" below. `Predator` (grazed) was dropped for now since
there's no real photo data for it yet; it can be added back later the moment
labeled examples exist, following the same fixed-folder pattern
(`config.py`'s `CLASS_NAMES` is the only place the fixed class list is
defined).

### Species scoping

The dataset directory is nested one level under a species slug
(`dataset/<species_slug>/...`) so a future second species can be added as a
sibling folder without restructuring anything. Today only
`kappaphycus_alvarezii` exists (`config.SPECIES_SLUG`) — this is a
placeholder, not a trained species classifier; the app currently identifies
only this one species. See the roadmap in `docs/STEP_BY_STEP.md` for when a
real species-identification model gets added.

### From raw class to category/condition/score

For the 5 fixed classes, labelers just assign one of the 5 names to a photo —
nothing changes about the labeling process itself. Internally,
`config.CLASS_TARGET_MAP` remaps each into what the model actually predicts: a
top-level **category** (`Healthy`/`Moderate`/`Low`), an optional **condition**
tag (`Dried`/`Decayed`/`Diseased`, only set when category != Healthy), a
heuristic **health-score anchor** in `[0, 10]`, and heuristic **dried%/decayed%
extent anchors** in `[0, 100]` (all used to train the corresponding
regression/classification heads):

| Raw folder | Category | Condition | Score anchor | Dried% anchor | Decayed% anchor |
|---|---|---|---|---|---|
| Healthy | Healthy | — | 9.0 | 0 | 0 |
| Moderate | Moderate | — | 6.0 | 0 | 5 |
| Low | Low | — | 3.0 | 0 | 10 |
| Decay | Low | Decayed | 2.0 | 5 | 70 |
| Dried | Low | Dried | 0.5 | 95 | 5 |

`Decay` and `Dried` always map to category **Low** — the descriptions below
(rot, zero living tissue) describe severe/advanced symptoms consistent with
the Low tier, with no ambiguity about severity. `Disease` is handled
differently — see "Disease folders" below, since unlike Decay/Dried, a
diseased specimen can be a small patch on an otherwise-Moderate specimen or
a widespread Low-severity case.

All anchors (score and dried%/decayed%) are a documented heuristic derived
from this guide's own severity language, not measured biological or
pixel-level ground truth — see `config.py`'s comments for the full
justification, and treat them as a first draft to revise once real
expert-scored/segmentation data exists.

Folder names must match exactly (case-sensitive) — they become the model's
class labels via `torchvision.datasets.ImageFolder`.

### Disease folders

Unlike the other symptom classes, disease severity isn't fixed — a small
diseased patch doesn't necessarily drag an otherwise-Moderate specimen down
to Low. So instead of a single flat `Disease/` folder, disease photos go into
folders named:

```
Disease_<Severity>              e.g. Disease_Moderate, Disease_Low
Disease_<Severity>_<Subtype>    e.g. Disease_Moderate_IceIce, Disease_Low_Bacterial
```

- `<Severity>` is **required** and must be exactly `Moderate` or `Low` — pick
  based on how much of the specimen is affected and how much overall decline
  is visible, same judgment call as choosing between the plain `Moderate` and
  `Low` classes.
- `<Subtype>` is **optional** (omit it, or use `Unknown`, if you can't
  identify the specific pathogen) and must be one of: `IceIce` (Ice-Ice
  Disease), `Epiphyte` (Epiphyte Infection), `Bacterial` (Bacterial Disease),
  or `Unknown`. This list is `config.DISEASE_SUBTYPE_NAMES` — extend it there
  if you need a new subtype.
- You do **not** need every severity/subtype combination to exist — only
  create the folders you actually have photos for.
  `python -m src.data.validate_dataset` (or `python scripts/split_dataset.py`)
  will raise a clear error if a `Disease_...` folder name doesn't parse (e.g.
  a typo'd severity or subtype), rather than silently mis-training on it.

Both `<Severity>` and `<Subtype>` are folder-encoded ground truth (not a
heuristic anchor) — the model's category/condition/disease-subtype heads are
trained directly against whatever you label the folder, same as the fixed
classes.

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

### Disease (`Disease_<Severity>[_<Subtype>]`)
- Visible lesions
- Infection symptoms (not explained by grazing or general decay)
- Choose `Moderate` severity for a small/contained diseased patch on an
  otherwise-decent specimen, `Low` severity for widespread infection with
  significant overall decline
- Subtype, if identifiable: `IceIce` (whitish, brittle lesions), `Epiphyte`
  (filamentous overgrowth on the surface), `Bacterial` (localized lesion with
  discoloration), or `Unknown`/omitted if the pathogen isn't clear

## Annotation practices

1. **One class per image.** If a specimen shows both decay and disease
   symptoms, pick the dominant/most obvious symptom, or better, exclude it
   from the Phase 1 dataset and flag it for a future combined-symptom model.
2. **Consistent framing.** Fill the frame with the specimen; avoid heavy
   background clutter that could let the model shortcut on non-biological
   cues (e.g. always photographing "Dried" samples on a particular boat
   deck).
3. **Two-annotator agreement for ambiguous classes** (Low vs. Moderate,
   Decay vs. Dried, Disease severity Moderate vs. Low) where possible.
   Disagreements should be discussed and resolved before the image enters
   the training set, not silently averaged.
4. **Do not relabel to fix class imbalance.** If a class is underrepresented,
   collect more images of that class — don't reclassify borderline images
   into it. `python -m src.data.validate_dataset` (see `ml/README.md`) flags
   underrepresented classes after every dataset update.
5. **Reject corrupt/blurry/out-of-focus images** rather than trying to force
   a label onto unusable data.
