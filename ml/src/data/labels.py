"""Single source of truth for what a class-folder name *means*.

Class folders are flat, ImageFolder-compatible leaf directories whose names
encode structured labels:

    <species_slug>_Healthy
    <species_slug>_Decay
    <species_slug>_Dried
    <species_slug>_Disease_<Severity>_<Subtype>[_<DiseaseName...>]
    Background                       (species-agnostic negative class)

Examples:
    Kappaphycus_alvarezii_Healthy
    Kappaphycus_alvarezii_Disease_Moderate_IceIce
    Kappaphycus_alvarezii_Disease_Low_Bacterial_Vibrio_sp
    Background

`parse_class_folder` turns a folder name into a ParsedLabel; `derive_targets`
turns a ParsedLabel into the multi-head training targets, applying the
heuristic anchors from config. Both the dataset loader, the retrain
materialization step, and the label-map builder go through here so the naming
convention is defined in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import (
    CONDITION_CLASSES,
    DECAYED_EXTENT_ANCHORS,
    DISEASE_SUBTYPES,
    DRIED_EXTENT_ANCHORS,
    HEALTH_SCORE_ANCHORS,
    SEVERITIES,
    SPECIES,
)

BACKGROUND = "Background"


@dataclass
class ParsedLabel:
    condition: str  # one of CONDITION_CLASSES
    severity: str | None = None  # Disease only, one of SEVERITIES
    subtype: str | None = None  # Disease only, one of DISEASE_SUBTYPES
    disease_name: str | None = None  # free-form, metadata only


class LabelParseError(ValueError):
    """Raised when a folder name doesn't fit the naming convention."""


def parse_class_folder(name: str, species_slug: str | None = None) -> ParsedLabel:
    """Parse a class-folder name into its structured label.

    Raises LabelParseError on anything that doesn't fit the convention, so
    dataset validation fails loudly rather than silently mislabeling.
    """
    species_slug = species_slug or SPECIES["slug"]

    if name == BACKGROUND:
        return ParsedLabel(condition=BACKGROUND)

    prefix = f"{species_slug}_"
    if not name.startswith(prefix):
        raise LabelParseError(
            f"Folder {name!r} is neither {BACKGROUND!r} nor prefixed with {prefix!r}."
        )
    remainder = name[len(prefix) :]
    parts = remainder.split("_")

    condition = parts[0]
    if condition not in CONDITION_CLASSES or condition == BACKGROUND:
        raise LabelParseError(
            f"Folder {name!r} has unknown condition {condition!r}; "
            f"expected one of {[c for c in CONDITION_CLASSES if c != BACKGROUND]}."
        )

    if condition != "Disease":
        if len(parts) > 1:
            raise LabelParseError(
                f"Folder {name!r}: condition {condition!r} takes no extra tokens."
            )
        return ParsedLabel(condition=condition)

    # Disease: <Severity>_<Subtype>[_<DiseaseName...>]
    if len(parts) < 3:
        raise LabelParseError(
            f"Folder {name!r}: Disease requires <Severity>_<Subtype>, "
            f"e.g. {species_slug}_Disease_Moderate_IceIce."
        )
    severity, subtype = parts[1], parts[2]
    if severity not in SEVERITIES:
        raise LabelParseError(f"Folder {name!r}: unknown severity {severity!r}; expected {SEVERITIES}.")
    if subtype not in DISEASE_SUBTYPES:
        raise LabelParseError(
            f"Folder {name!r}: unknown subtype {subtype!r}; expected {DISEASE_SUBTYPES}."
        )
    disease_name = "_".join(parts[3:]) or None
    return ParsedLabel(condition="Disease", severity=severity, subtype=subtype, disease_name=disease_name)


def build_class_folder(
    condition: str,
    severity: str | None = None,
    subtype: str | None = None,
    disease_name: str | None = None,
    species_slug: str | None = None,
) -> str:
    """Inverse of parse_class_folder — build a canonical folder name from
    structured fields. Used by the retrain step when materializing DB rows."""
    if condition == BACKGROUND:
        return BACKGROUND
    species_slug = species_slug or SPECIES["slug"]
    if condition == "Disease":
        if severity not in SEVERITIES or subtype not in DISEASE_SUBTYPES:
            raise LabelParseError(
                f"Disease requires severity in {SEVERITIES} and subtype in {DISEASE_SUBTYPES}."
            )
        tokens = [species_slug, "Disease", severity, subtype]
        if disease_name:
            tokens.append(disease_name)
        return "_".join(tokens)
    return f"{species_slug}_{condition}"


def health_level(condition: str, severity: str | None) -> str | None:
    """Discrete display level from the folder's structured label. Background
    has no level. Disease uses the folder's severity token directly (at
    inference this is re-derived from the regressed score instead)."""
    if condition == BACKGROUND:
        return None
    if condition == "Healthy":
        return "Healthy"
    if condition in ("Decay", "Dried"):
        return "Low"
    if condition == "Disease":
        return severity  # "Moderate" or "Low"
    return None


def derive_targets(parsed: ParsedLabel) -> dict:
    """Turn a ParsedLabel into the multi-head training targets, applying the
    heuristic anchors. Masks tell the loss which heads to supervise:
      - health_mask/extent_mask = 0 for Background (nothing to regress)
      - subtype_mask = 1 only for Disease
    """
    condition = parsed.condition
    condition_id = CONDITION_CLASSES.index(condition)
    is_background = condition == BACKGROUND
    is_disease = condition == "Disease"

    if is_disease:
        score_key = f"Disease:{parsed.severity}"
    else:
        score_key = condition
    health_score = HEALTH_SCORE_ANCHORS.get(score_key, 0.0)

    subtype_id = DISEASE_SUBTYPES.index(parsed.subtype) if is_disease and parsed.subtype else 0

    return {
        "condition_id": condition_id,
        "health_score": 0.0 if is_background else float(health_score),
        "subtype_id": subtype_id,
        "dried_extent": 0.0 if is_background else float(DRIED_EXTENT_ANCHORS.get(condition, 0.0)),
        "decayed_extent": 0.0 if is_background else float(DECAYED_EXTENT_ANCHORS.get(condition, 0.0)),
        "subtype_mask": 1.0 if is_disease else 0.0,
        "health_mask": 0.0 if is_background else 1.0,
        "extent_mask": 0.0 if is_background else 1.0,
    }
