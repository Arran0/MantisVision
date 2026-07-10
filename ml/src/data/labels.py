"""Single source of truth for what a class-folder name *means*.

Class folders are flat, ImageFolder-compatible leaf directories whose names
encode structured labels. Severity comes BEFORE the condition token:

    <species_slug>_Healthy
    <species_slug>_Low_Decay                                  (Decay is always "Low" severity)
    <species_slug>_Low_Dried                                  (Dried is always "Low" severity)
    <species_slug>_<Severity>_Disease_<Subtype>[_<DiseaseName...>]
    Background                                                 (species-agnostic negative class)

Examples:
    Kappaphycus_alvarezii_Healthy
    Kappaphycus_alvarezii_Low_Decay
    Kappaphycus_alvarezii_Low_Dried
    Kappaphycus_alvarezii_Moderate_Disease_IceIce
    Kappaphycus_alvarezii_Low_Disease_Bacterial_Vibrio_sp
    Background

Decay and Dried only ever take "Low" severity — there is no Moderate or
Healthy-severity bucket for them (see config.FIXED_SEVERITY_CONDITIONS). The
severity token is still present in the folder name (not omitted) so every
non-Healthy, non-Background folder has the same `<Severity>_<Condition>...`
shape.

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
    FIXED_SEVERITY_CONDITIONS,
    HEALTH_SCORE_ANCHORS,
    SEVERITIES,
    SPECIES,
)

BACKGROUND = "Background"


@dataclass
class ParsedLabel:
    condition: str  # one of CONDITION_CLASSES
    severity: str | None = None  # None only for Healthy/Background
    subtype: str | None = None  # Disease only, one of DISEASE_SUBTYPES
    disease_name: str | None = None  # Disease only, free-form, metadata only


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
    parts = name[len(prefix) :].split("_")

    if parts[0] == "Healthy":
        if len(parts) > 1:
            raise LabelParseError(f"Folder {name!r}: 'Healthy' takes no extra tokens.")
        return ParsedLabel(condition="Healthy")

    if parts[0] not in SEVERITIES:
        raise LabelParseError(
            f"Folder {name!r}: expected 'Healthy' or a severity prefix {SEVERITIES}, got {parts[0]!r}."
        )
    severity = parts[0]
    if len(parts) < 2:
        raise LabelParseError(
            f"Folder {name!r}: severity {severity!r} must be followed by Decay, Dried, or Disease."
        )
    condition = parts[1]

    if condition in FIXED_SEVERITY_CONDITIONS:
        required = FIXED_SEVERITY_CONDITIONS[condition]
        if severity != required:
            raise LabelParseError(
                f"Folder {name!r}: {condition} only supports {required!r} severity, got {severity!r}."
            )
        if len(parts) > 2:
            raise LabelParseError(f"Folder {name!r}: {condition} takes no extra tokens.")
        return ParsedLabel(condition=condition, severity=severity)

    if condition == "Disease":
        if len(parts) < 3:
            raise LabelParseError(
                f"Folder {name!r}: Disease requires <Severity>_Disease_<Subtype>, "
                f"e.g. {species_slug}_Moderate_Disease_IceIce."
            )
        subtype = parts[2]
        if subtype not in DISEASE_SUBTYPES:
            raise LabelParseError(
                f"Folder {name!r}: unknown subtype {subtype!r}; expected {DISEASE_SUBTYPES}."
            )
        disease_name = "_".join(parts[3:]) or None
        return ParsedLabel(condition="Disease", severity=severity, subtype=subtype, disease_name=disease_name)

    raise LabelParseError(
        f"Folder {name!r}: unrecognized condition token {condition!r} after severity {severity!r}; "
        f"expected Decay, Dried, or Disease."
    )


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

    if condition == "Healthy":
        return f"{species_slug}_Healthy"

    if condition in FIXED_SEVERITY_CONDITIONS:
        return f"{species_slug}_{FIXED_SEVERITY_CONDITIONS[condition]}_{condition}"

    if condition == "Disease":
        if severity not in SEVERITIES or subtype not in DISEASE_SUBTYPES:
            raise LabelParseError(
                f"Disease requires severity in {SEVERITIES} and subtype in {DISEASE_SUBTYPES}."
            )
        tokens = [species_slug, severity, "Disease", subtype]
        if disease_name:
            tokens.append(disease_name)
        return "_".join(tokens)

    raise LabelParseError(f"Unknown condition {condition!r}.")


def health_level(condition: str, severity: str | None) -> str | None:
    """Discrete display level from the folder's structured label. Background
    has no level; Healthy is its own level. Decay/Dried/Disease all carry an
    explicit severity token now, so the level is just that severity (at
    inference, Disease's severity is re-derived from the regressed score
    instead of trusted as-is)."""
    if condition == BACKGROUND:
        return None
    if condition == "Healthy":
        return "Healthy"
    return severity  # Decay/Dried -> "Low"; Disease -> "Moderate" or "Low"


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
