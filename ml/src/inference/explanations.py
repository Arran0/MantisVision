"""Preset bullet-point explanations + recommendations, composed from
(category, condition, disease_subtype, dried%/decayed% extent).

This is still preset/rule-based text (a placeholder for real NLG/LLM-based,
per-image explanations later) — but restructured from one fixed sentence per
flat class into a bullet list built from a handful of small tables, so e.g. a
Low+Diseased+IceIce result shows the Low symptoms, the general disease
symptoms, the Ice-Ice-specific symptoms, and the extent percentages, without
duplicating content across a giant combinatorial lookup. Category/condition
bullet content is drawn directly from docs/DATASET_LABELING_GUIDE.md's class
definitions; disease-subtype content matches the subtypes already named in
docs/STEP_BY_STEP.md's roadmap. Extent bullets are generated from the
model's dried%/decayed% output, which is itself a documented heuristic (see
config.py) — not a measured percentage.
"""
from __future__ import annotations

CATEGORY_BULLETS: dict[str, list[str]] = {
    "Healthy": [
        "Bright green coloration across the frame",
        "No whitening or bleaching detected",
        "No broken or damaged branches observed",
    ],
    "Moderate": [
        "Slight whitening visible on branch tips",
        "Small areas of tissue loss present",
        "Thallus still appears to be actively growing",
    ],
    "Low": [
        "Significant discoloration compared to healthy tissue",
        "Reduced branching density observed",
        "Overall structure visibly diminished",
    ],
}

CONDITION_BULLETS: dict[str, list[str]] = {
    "Dried": [
        "Tissue appears completely dried out and bleached white",
        "Specimen appears detached from the line/substrate",
        "No living (green/pigmented) tissue visible anywhere in frame",
    ],
    "Decayed": [
        "Signs of tissue melting visible",
        "Brown patches present on the thallus",
        "Pattern consistent with rot rather than simple discoloration",
    ],
    "Diseased": [
        "Visible lesions on the tissue surface",
        "Symptoms consistent with infection rather than grazing damage",
        "Pattern doesn't match typical decay or drying",
    ],
}

# Disease-subtype content — the taxonomy from docs/STEP_BY_STEP.md's roadmap.
# "Unknown" is the fallback when a Diseased sample's specific pathogen hasn't
# been identified (either no subtype-labeled training data yet, or the model
# genuinely isn't sure).
DISEASE_SUBTYPE_DISPLAY_NAMES: dict[str, str] = {
    "Unknown": "Unidentified pathogen",
    "IceIce": "Ice-Ice Disease",
    "Epiphyte": "Epiphyte Infection",
    "Bacterial": "Bacterial Disease",
}

DISEASE_SUBTYPE_BULLETS: dict[str, list[str]] = {
    "IceIce": [
        "Whitish, brittle lesions typical of Ice-Ice disease",
        "Tissue softening near the lesion site",
    ],
    "Epiphyte": [
        "Fine filamentous growth on the thallus surface",
        "Pattern consistent with epiphyte overgrowth rather than internal infection",
    ],
    "Bacterial": [
        "Lesion pattern consistent with a bacterial infection",
        "Localized discoloration around the affected site",
    ],
    "Unknown": [
        "Disease symptoms present but the specific pathogen could not be determined",
    ],
}

CATEGORY_RECOMMENDATIONS: dict[str, str] = {
    "Healthy": "Continue routine monitoring. No action needed.",
    "Moderate": "Increase water movement. Inspect for grazers and early disease signs.",
    "Low": "Relocate to better water flow if possible. Increase inspection frequency.",
}

CONDITION_RECOMMENDATIONS: dict[str, str] = {
    "Dried": "Remove and dispose of dried-out material. Inspect surrounding line for early-stage damage.",
    "Decayed": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
    "Diseased": "Isolate affected line segments if the condition worsens.",
}

DISEASE_SUBTYPE_RECOMMENDATIONS: dict[str, str] = {
    "IceIce": "Consistent with Ice-Ice disease — check for temperature/salinity shifts and reduce handling stress.",
    "Epiphyte": "Consistent with epiphyte infection — clean the line and reduce nutrient loading if possible.",
    "Bacterial": "Consistent with a bacterial infection — isolate affected line segments and monitor closely.",
    "Unknown": "Consult a specialist to confirm the pathogen before treating.",
}

# Extent bullets are only added above this threshold (percentage points) so a
# small heuristic-regression wobble on an otherwise-healthy image doesn't
# produce a spurious "X% dried" bullet.
EXTENT_BULLET_THRESHOLD = 5.0


def _extent_bullets(dried_pct: float, decayed_pct: float) -> list[str]:
    bullets = []
    if dried_pct >= EXTENT_BULLET_THRESHOLD:
        bullets.append(f"Approximately {dried_pct:.0f}% of visible tissue appears dried")
    if decayed_pct >= EXTENT_BULLET_THRESHOLD:
        bullets.append(f"Approximately {decayed_pct:.0f}% of visible tissue appears decayed")
    return bullets


def explanation_bullets_for(
    category: str,
    condition: str | None,
    disease_subtype: str | None = None,
    dried_pct: float = 0.0,
    decayed_pct: float = 0.0,
) -> list[str]:
    bullets = list(CATEGORY_BULLETS.get(category, ["No explanation available for this category yet."]))
    if condition:
        bullets += CONDITION_BULLETS.get(condition, [])
    if condition == "Diseased" and disease_subtype:
        bullets += DISEASE_SUBTYPE_BULLETS.get(disease_subtype, [])
    bullets += _extent_bullets(dried_pct, decayed_pct)
    return bullets


def recommendation_for(category: str, condition: str | None, disease_subtype: str | None = None) -> str:
    parts = [CATEGORY_RECOMMENDATIONS.get(category, "No recommendation available for this category yet.")]
    if condition:
        condition_text = CONDITION_RECOMMENDATIONS.get(condition)
        if condition_text:
            parts.append(condition_text)
    if condition == "Diseased" and disease_subtype:
        subtype_text = DISEASE_SUBTYPE_RECOMMENDATIONS.get(disease_subtype)
        if subtype_text:
            parts.append(subtype_text)
    return " ".join(parts)
