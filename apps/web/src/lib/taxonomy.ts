// The dataset taxonomy — active species, condition "buckets", severities,
// disease subtypes, the heuristic regression anchors, and the preset
// explanation/recommendation copy. This used to be hardcoded here (and in
// ml/config.py + ml/src/inference/explanations.py); it is now an admin-editable
// JSONB document stored in Supabase (table dataset_taxonomy). This module holds
// the shared TypeScript shape, the DEFAULT_TAXONOMY fallback (kept in sync with
// the SQL seed in supabase/migrations/20260714000005_dataset_taxonomy.sql and
// ml/config.py's fallbacks), plus validation/derivation helpers usable from
// both server and client code.

export interface SpeciesDef {
  name: string;
  slug: string;
}

export interface ConditionDef {
  name: string;
  is_background: boolean;
  // For conditions pinned to one severity (e.g. Decay/Dried are always "Low").
  // null means the condition uses the free severity list (or none, for Healthy).
  fixed_severity: string | null;
  // true for disease-like conditions that carry a subtype (e.g. Disease).
  requires_subtype: boolean;
  // Heuristic training anchors (0–100). health_score_anchor is used unless the
  // condition is severity-split, in which case health_score_anchors_by_severity
  // is keyed by severity name.
  health_score_anchor: number | null;
  health_score_anchors_by_severity: Record<string, number>;
  dried_extent_anchor: number;
  decayed_extent_anchor: number;
  // Preset copy shown to end users for a prediction of this condition.
  explanation: string;
  recommendation: string;
}

export interface SubtypeDef {
  name: string;
  note: string;
}

export interface TaxonomyDoc {
  species: SpeciesDef[];
  active_species_slug: string;
  severities: string[];
  disease_moderate_min: number;
  conditions: ConditionDef[];
  disease_subtypes: SubtypeDef[];
}

// Fallback used when no taxonomy row exists yet. Mirrors the SQL seed.
export const DEFAULT_TAXONOMY: TaxonomyDoc = {
  species: [{ name: "Kappaphycus alvarezii", slug: "Kappaphycus_alvarezii" }],
  active_species_slug: "Kappaphycus_alvarezii",
  severities: ["Moderate", "Low"],
  disease_moderate_min: 45.0,
  conditions: [
    {
      name: "Background",
      is_background: true,
      fixed_severity: null,
      requires_subtype: false,
      health_score_anchor: null,
      health_score_anchors_by_severity: {},
      dried_extent_anchor: 0.0,
      decayed_extent_anchor: 0.0,
      explanation: "No seaweed specimen was detected in this image.",
      recommendation: "Point the camera at a seaweed specimen, filling the frame, and try again.",
    },
    {
      name: "Healthy",
      is_background: false,
      fixed_severity: null,
      requires_subtype: false,
      health_score_anchor: 90.0,
      health_score_anchors_by_severity: {},
      dried_extent_anchor: 0.0,
      decayed_extent_anchor: 0.0,
      explanation:
        "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
      recommendation: "Continue routine monitoring. No action needed.",
    },
    {
      name: "Disease",
      is_background: false,
      fixed_severity: null,
      requires_subtype: true,
      health_score_anchor: null,
      health_score_anchors_by_severity: { Moderate: 60.0, Low: 30.0 },
      dried_extent_anchor: 0.0,
      decayed_extent_anchor: 20.0,
      explanation:
        "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
      recommendation: "Isolate affected line segments and confirm the pathogen before treating.",
    },
    {
      name: "Decay",
      is_background: false,
      fixed_severity: "Low",
      requires_subtype: false,
      health_score_anchor: 20.0,
      health_score_anchors_by_severity: {},
      dried_extent_anchor: 10.0,
      decayed_extent_anchor: 80.0,
      explanation:
        "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
      recommendation:
        "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
    },
    {
      name: "Dried",
      is_background: false,
      fixed_severity: "Low",
      requires_subtype: false,
      health_score_anchor: 5.0,
      health_score_anchors_by_severity: {},
      dried_extent_anchor: 90.0,
      decayed_extent_anchor: 0.0,
      explanation: "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
      recommendation: "Remove and dispose of dried-out material. Inspect the surrounding line for early damage.",
    },
  ],
  disease_subtypes: [
    {
      name: "IceIce",
      note: "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity.",
    },
    { name: "Epiphyte", note: "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow." },
    { name: "Bacterial", note: "Possible bacterial infection: isolate and consult a specialist before any treatment." },
    { name: "Bleaching", note: "Bleaching suspected: check for temperature/light stress and relocate if possible." },
    { name: "Unknown", note: "Subtype unclear: photograph affected areas closely and consult a specialist." },
  ],
};

// --- Derivation helpers ----------------------------------------------------

export function conditionNames(doc: TaxonomyDoc): string[] {
  return doc.conditions.map((c) => c.name);
}

export function subtypeNames(doc: TaxonomyDoc): string[] {
  return doc.disease_subtypes.map((s) => s.name);
}

export function findCondition(doc: TaxonomyDoc, name: string): ConditionDef | undefined {
  return doc.conditions.find((c) => c.name === name);
}

export function isConditionName(doc: TaxonomyDoc, value: string): boolean {
  return doc.conditions.some((c) => c.name === value);
}

export function isSeverityName(doc: TaxonomyDoc, value: string): boolean {
  return doc.severities.includes(value);
}

export function isSubtypeName(doc: TaxonomyDoc, value: string): boolean {
  return doc.disease_subtypes.some((s) => s.name === value);
}

// Backward-compatible constant exports derived from the default taxonomy, for
// any consumer that only needs the built-in set (e.g. initial UI state before
// the live taxonomy loads).
export const CONDITIONS = conditionNames(DEFAULT_TAXONOMY);
export const SEVERITIES = DEFAULT_TAXONOMY.severities;
export const DISEASE_SUBTYPES = subtypeNames(DEFAULT_TAXONOMY);

// --- Validation ------------------------------------------------------------

const SLUG_RE = /^[A-Za-z0-9_]+$/;
const TOKEN_RE = /^[A-Za-z0-9]+$/; // condition/severity/subtype tokens feed folder names

function isFraction(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 100;
}

// Validates a candidate taxonomy document. Returns null if valid, else a
// human-readable reason. The rules mirror the invariants ml/config.py and the
// folder-naming convention (ml/src/data/labels.py) rely on.
export function validateTaxonomy(doc: unknown): string | null {
  if (typeof doc !== "object" || doc === null) return "Taxonomy must be an object.";
  const t = doc as Partial<TaxonomyDoc>;

  if (!Array.isArray(t.species) || t.species.length === 0) return "At least one species is required.";
  for (const s of t.species) {
    if (!s?.name?.trim()) return "Every species needs a name.";
    if (!s?.slug || !SLUG_RE.test(s.slug))
      return `Species slug ${JSON.stringify(s?.slug)} must contain only letters, numbers, and underscores.`;
  }
  const slugs = t.species.map((s) => s.slug);
  if (new Set(slugs).size !== slugs.length) return "Species slugs must be unique.";
  if (!t.active_species_slug || !slugs.includes(t.active_species_slug))
    return "active_species_slug must match one of the species.";

  if (!Array.isArray(t.severities) || t.severities.length === 0) return "At least one severity is required.";
  for (const sev of t.severities) {
    if (!TOKEN_RE.test(sev)) return `Severity ${JSON.stringify(sev)} must be a single alphanumeric token.`;
  }
  if (new Set(t.severities).size !== t.severities.length) return "Severities must be unique.";

  if (!isFraction(t.disease_moderate_min)) return "disease_moderate_min must be a number between 0 and 100.";

  if (!Array.isArray(t.conditions) || t.conditions.length === 0) return "At least one condition is required.";
  const condNames = t.conditions.map((c) => c.name);
  if (new Set(condNames).size !== condNames.length) return "Condition names must be unique.";
  const backgrounds = t.conditions.filter((c) => c.is_background);
  if (backgrounds.length !== 1) return "Exactly one condition must be marked as the Background (no-seaweed) class.";

  const subtypeSet = new Set((t.disease_subtypes ?? []).map((s) => s.name));
  for (const c of t.conditions) {
    if (!c.name || !TOKEN_RE.test(c.name))
      return `Condition ${JSON.stringify(c.name)} must be a single alphanumeric token (folder-name safe).`;
    if (c.fixed_severity !== null && !t.severities.includes(c.fixed_severity))
      return `Condition ${c.name}: fixed_severity ${JSON.stringify(c.fixed_severity)} is not in the severity list.`;
    if (c.health_score_anchor !== null && !isFraction(c.health_score_anchor))
      return `Condition ${c.name}: health_score_anchor must be null or between 0 and 100.`;
    if (!isFraction(c.dried_extent_anchor)) return `Condition ${c.name}: dried_extent_anchor must be between 0 and 100.`;
    if (!isFraction(c.decayed_extent_anchor))
      return `Condition ${c.name}: decayed_extent_anchor must be between 0 and 100.`;
    if (c.requires_subtype) {
      const perSev = c.health_score_anchors_by_severity ?? {};
      for (const sev of t.severities) {
        if (!isFraction(perSev[sev]))
          return `Condition ${c.name} carries a subtype, so it needs a health score anchor (0–100) for severity "${sev}".`;
      }
    }
  }
  // A subtype-carrying condition is only meaningful if subtypes exist.
  if (t.conditions.some((c) => c.requires_subtype) && subtypeSet.size === 0)
    return "A condition requires a subtype, but no disease subtypes are defined.";

  for (const s of t.disease_subtypes ?? []) {
    if (!s.name || !TOKEN_RE.test(s.name))
      return `Disease subtype ${JSON.stringify(s.name)} must be a single alphanumeric token.`;
  }
  if (subtypeSet.size !== (t.disease_subtypes ?? []).length) return "Disease subtype names must be unique.";

  return null;
}
