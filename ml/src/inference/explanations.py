"""Rule-based explanation + recommendation text per health class.

This is a placeholder for real NLG/LLM-based explanations later. For the MVP
it gives the user something actionable immediately, matching the example
output in the spec (species / health / confidence / explanation /
recommendation).
"""
from __future__ import annotations

EXPLANATIONS: dict[str, str] = {
    "Healthy": "Bright green coloration with no whitening or broken branches detected.",
    "Moderate": "Minor bleaching on branches and early tissue degradation observed.",
    "Low": "Significant discoloration and reduced branching compared to healthy tissue.",
    "Decay": "Signs of tissue melting, brown patches, and rot are visible.",
    "Dried": "Tissue is completely dried out/detached with no living tissue remaining.",
    "Disease": "Lesions and infection symptoms consistent with a disease outbreak.",
}

RECOMMENDATIONS: dict[str, str] = {
    "Healthy": "Continue routine monitoring. No action needed.",
    "Moderate": "Increase water movement. Inspect for grazers and early disease signs.",
    "Low": "Relocate to better water flow if possible. Increase inspection frequency.",
    "Decay": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
    "Dried": "Remove and dispose of dried-out material. Inspect surrounding line for early-stage damage.",
    "Disease": "Isolate affected line segments. Consult a specialist to confirm the pathogen (see Disease Model, future work).",
}


def explanation_for(label: str) -> str:
    return EXPLANATIONS.get(label, "No explanation available for this class yet.")


def recommendation_for(label: str) -> str:
    return RECOMMENDATIONS.get(label, "No recommendation available for this class yet.")
