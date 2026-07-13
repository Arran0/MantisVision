"""Rule-based explanation + recommendation text, keyed off the predicted
condition (and, for Disease, the subtype and derived level).

This is a placeholder for real NLG/LLM-based explanations later. For now it
gives the user something actionable immediately.
"""
from __future__ import annotations

CONDITION_EXPLANATIONS: dict[str, str] = {
    "Background": "No seaweed specimen was detected in this image.",
    # Kappaphycus alvarezii is healthy at green, brown, or yellow-brown shades
    # alike — the tell is vivid, even colour and intact branching, not any one
    # specific hue.
    "Healthy": "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
    "Decay": "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
    "Dried": "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
    "Disease": "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
}

CONDITION_RECOMMENDATIONS: dict[str, str] = {
    "Background": "Point the camera at a seaweed specimen, filling the frame, and try again.",
    "Healthy": "Continue routine monitoring. No action needed.",
    "Decay": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
    "Dried": "Remove and dispose of dried-out material. Inspect the surrounding line for early damage.",
    "Disease": "Isolate affected line segments and confirm the pathogen before treating.",
}

# Extra, subtype-specific guidance appended when the condition is Disease.
SUBTYPE_NOTES: dict[str, str] = {
    "IceIce": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity.",
    "Epiphyte": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow.",
    "Bacterial": "Possible bacterial infection: isolate and consult a specialist before any treatment.",
    "Bleaching": "Bleaching suspected: check for temperature/light stress and relocate if possible.",
    "Unknown": "Subtype unclear: photograph affected areas closely and consult a specialist.",
}


def explanation_for(condition: str, subtype: str | None = None, level: str | None = None) -> str:
    base = CONDITION_EXPLANATIONS.get(condition, "No explanation available for this condition yet.")
    if condition == "Disease" and level:
        base = f"{base} Severity assessed as {level.lower()}."
    return base


def recommendation_for(condition: str, subtype: str | None = None) -> str:
    base = CONDITION_RECOMMENDATIONS.get(condition, "No recommendation available for this condition yet.")
    if condition == "Disease" and subtype:
        note = SUBTYPE_NOTES.get(subtype)
        if note:
            return f"{base} {note}"
    return base
