"""Preset bullet-point explanations + recommendations, composed from
(category, condition).

This is still preset/rule-based text (a placeholder for real NLG/LLM-based,
per-image explanations later) — but restructured from one fixed sentence per
flat class into a bullet list built from two small tables, so a Low+Dried
result shows both the general "Low" symptoms and the "Dried"-specific ones,
without duplicating content across a 3x4 lookup. All bullet content is drawn
directly from docs/DATASET_LABELING_GUIDE.md's class definitions — nothing
invented beyond what that guide already documents.
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

CATEGORY_RECOMMENDATIONS: dict[str, str] = {
    "Healthy": "Continue routine monitoring. No action needed.",
    "Moderate": "Increase water movement. Inspect for grazers and early disease signs.",
    "Low": "Relocate to better water flow if possible. Increase inspection frequency.",
}

CONDITION_RECOMMENDATIONS: dict[str, str] = {
    "Dried": "Remove and dispose of dried-out material. Inspect surrounding line for early-stage damage.",
    "Decayed": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
    "Diseased": "Isolate affected line segments. Consult a specialist to confirm the pathogen (see Disease Model, future work).",
}


def explanation_bullets_for(category: str, condition: str | None) -> list[str]:
    bullets = list(CATEGORY_BULLETS.get(category, ["No explanation available for this category yet."]))
    if condition:
        bullets += CONDITION_BULLETS.get(condition, [])
    return bullets


def recommendation_for(category: str, condition: str | None) -> str:
    base = CATEGORY_RECOMMENDATIONS.get(category, "No recommendation available for this category yet.")
    if not condition:
        return base
    condition_text = CONDITION_RECOMMENDATIONS.get(condition)
    return f"{base} {condition_text}" if condition_text else base
