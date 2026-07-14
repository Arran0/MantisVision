// The measurement schema — the admin-editable definition of every per-image
// measurement the model predicts (classification, regression, or
// segmentation), plus the active species and preset explanation/
// recommendation copy per class. This used to be a fixed condition/severity/
// disease-subtype taxonomy hardcoded across ml/config.py,
// ml/src/inference/explanations.py, and this file; it is now a versioned
// JSONB document stored in Supabase (table measurement_schema), so an admin
// can add a whole new measurement (e.g. "moisture", "gel_strength",
// "biofouling") without any code change. New measurements with no
// ground-truth data yet simply stay masked (untrained) until values arrive.
//
// This module holds the shared TypeScript shape, the DEFAULT_SCHEMA fallback
// (kept in sync with the SQL seed in
// supabase/migrations/20260714000005_measurement_schema.sql and
// ml/config.py's fallback), plus validation/derivation helpers usable from
// both server and client code.

export type MeasurementType = "classification" | "regression" | "segmentation";

export interface SpeciesDef {
  name: string;
  slug: string;
}

// A class of a classification measurement. `explanation`/`recommendation` are
// preset copy shown to end users for a prediction of this class (used mainly
// by the primary "condition"-like measurement); `note` is extra copy appended
// onto a *parent* measurement's recommendation when this class is predicted
// (used by e.g. disease_subtype's per-subtype guidance).
export interface ClassDef {
  name: string;
  explanation?: string;
  recommendation?: string;
  note?: string;
}

export interface SegClassDef {
  name: string;
  color: string; // hex, e.g. "#22c55e" — used for the mask legend/overlay
}

// Gates a measurement so it's only meaningful (supervised in training, shown
// in the UI) when another classification measurement's value matches. E.g.
// disease_subtype only applies_when condition equals "Disease"; health_score
// applies_when condition not_equals "Background". Exactly one of
// equals/not_equals should be set.
export interface AppliesWhen {
  key: string;
  equals?: string;
  not_equals?: string;
}

export interface MeasurementDef {
  key: string; // stable identifier (snake_case), unique, used as the JSON/manifest key
  label: string;
  type: MeasurementType;
  loss_weight: number;
  applies_when?: AppliesWhen | null;
  // classification only
  background_class?: string | null; // name of the "no subject" class, if any
  classes?: ClassDef[];
  // regression only
  unit?: string;
  min?: number;
  max?: number;
  // segmentation only
  seg_classes?: SegClassDef[];
}

export interface SchemaDoc {
  species: SpeciesDef[];
  active_species_slug: string;
  // A regression-derived display threshold: a classification predicted as
  // "Disease" (or whichever class conventionally carries a Moderate/Low
  // split) shows as "Moderate" when health_score is at or above this, else
  // "Low". Kept as a single scalar for now — see ml/src/inference/predictor.py.
  disease_moderate_min: number;
  measurements: MeasurementDef[];
}

// Fallback used when no schema row exists yet. Mirrors the SQL seed.
export const DEFAULT_SCHEMA: SchemaDoc = {
  species: [{ name: "Kappaphycus alvarezii", slug: "Kappaphycus_alvarezii" }],
  active_species_slug: "Kappaphycus_alvarezii",
  disease_moderate_min: 45.0,
  measurements: [
    {
      key: "condition",
      label: "Condition",
      type: "classification",
      loss_weight: 1.0,
      background_class: "Background",
      classes: [
        {
          name: "Background",
          explanation: "No seaweed specimen was detected in this image.",
          recommendation: "Point the camera at a seaweed specimen, filling the frame, and try again.",
        },
        {
          name: "Healthy",
          explanation:
            "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
          recommendation: "Continue routine monitoring. No action needed.",
        },
        {
          name: "Disease",
          explanation:
            "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
          recommendation: "Isolate affected line segments and confirm the pathogen before treating.",
        },
        {
          name: "Decay",
          explanation:
            "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
          recommendation:
            "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
        },
        {
          name: "Dried",
          explanation: "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
          recommendation: "Remove and dispose of dried-out material. Inspect the surrounding line for early damage.",
        },
      ],
    },
    {
      key: "disease_subtype",
      label: "Disease subtype",
      type: "classification",
      loss_weight: 0.5,
      applies_when: { key: "condition", equals: "Disease" },
      classes: [
        {
          name: "IceIce",
          note: "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity.",
        },
        { name: "Epiphyte", note: "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow." },
        { name: "Bacterial", note: "Possible bacterial infection: isolate and consult a specialist before any treatment." },
        { name: "Bleaching", note: "Bleaching suspected: check for temperature/light stress and relocate if possible." },
        { name: "Unknown", note: "Subtype unclear: photograph affected areas closely and consult a specialist." },
      ],
    },
    {
      key: "health_score",
      label: "Health score",
      type: "regression",
      loss_weight: 1.0,
      unit: "score",
      min: 0,
      max: 100,
      applies_when: { key: "condition", not_equals: "Background" },
    },
    {
      key: "dried_extent",
      label: "Dried extent",
      type: "regression",
      loss_weight: 0.5,
      unit: "pct",
      min: 0,
      max: 100,
      applies_when: { key: "condition", not_equals: "Background" },
    },
    {
      key: "decayed_extent",
      label: "Decayed extent",
      type: "regression",
      loss_weight: 0.5,
      unit: "pct",
      min: 0,
      max: 100,
      applies_when: { key: "condition", not_equals: "Background" },
    },
  ],
};

// --- Derivation helpers ----------------------------------------------------

export function findMeasurement(doc: SchemaDoc, key: string): MeasurementDef | undefined {
  return doc.measurements.find((m) => m.key === key);
}

export function classificationMeasurements(doc: SchemaDoc): MeasurementDef[] {
  return doc.measurements.filter((m) => m.type === "classification");
}

// The measurement that flags "no subject in frame" (analogous to the old
// Background condition). Picks the first classification measurement that
// declares a background_class, if any.
export function getPrimaryClassification(doc: SchemaDoc): MeasurementDef | undefined {
  return doc.measurements.find((m) => m.type === "classification" && m.background_class);
}

export function classNames(measurement: MeasurementDef): string[] {
  return (measurement.classes ?? []).map((c) => c.name);
}

// Whether `measurement` is active given the current values of *other*
// measurements (keyed by measurement key -> chosen class name). A measurement
// with no applies_when is always active. Values are read as unknown so this
// works with both the client's Record<string,string> class-value state and
// the server's parsed-JSON measurements payload.
export function measurementApplies(measurement: MeasurementDef, values: Record<string, unknown>): boolean {
  const cond = measurement.applies_when;
  if (!cond) return true;
  const parentValue = values[cond.key];
  if (parentValue === undefined || parentValue === null) return false;
  if (cond.equals !== undefined) return parentValue === cond.equals;
  if (cond.not_equals !== undefined) return parentValue !== cond.not_equals;
  return true;
}

// Whether `value` is a legal value for `measurement` (ignoring applies_when —
// callers check that separately since it depends on sibling measurements).
export function isValueValidForMeasurement(measurement: MeasurementDef, value: unknown): boolean {
  if (measurement.type === "classification") {
    return typeof value === "string" && classNames(measurement).includes(value);
  }
  if (measurement.type === "regression") {
    const min = measurement.min ?? 0;
    const max = measurement.max ?? 100;
    return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max;
  }
  if (measurement.type === "segmentation") {
    // A segmentation value is a training-masks storage path.
    return typeof value === "string" && value.length > 0;
  }
  return false;
}

// --- Validation ------------------------------------------------------------

const SLUG_RE = /^[A-Za-z0-9_]+$/;
const KEY_RE = /^[a-z][a-z0-9_]*$/; // measurement keys are manifest/JSON identifiers
const TOKEN_RE = /^[A-Za-z0-9]+$/; // class/seg-class names stay folder-name safe
const COLOR_RE = /^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$/;

function isFraction(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 100;
}

// Validates a candidate schema document. Returns null if valid, else a
// human-readable reason. The rules mirror the invariants the ML pipeline
// (ml/config.py, ml/src/data/annotations.py) relies on.
export function validateSchema(doc: unknown): string | null {
  if (typeof doc !== "object" || doc === null) return "Schema must be an object.";
  const t = doc as Partial<SchemaDoc>;

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

  if (!isFraction(t.disease_moderate_min)) return "disease_moderate_min must be a number between 0 and 100.";

  if (!Array.isArray(t.measurements) || t.measurements.length === 0)
    return "At least one measurement is required.";

  const keys = t.measurements.map((m) => m.key);
  if (new Set(keys).size !== keys.length) return "Measurement keys must be unique.";

  for (const m of t.measurements) {
    if (!m.key || !KEY_RE.test(m.key))
      return `Measurement key ${JSON.stringify(m.key)} must be lowercase snake_case (e.g. "health_score").`;
    if (m.key === "species") return `Measurement key "species" is reserved.`;
    if (!m.label?.trim()) return `Measurement ${m.key}: label is required.`;
    if (!["classification", "regression", "segmentation"].includes(m.type))
      return `Measurement ${m.key}: type must be classification, regression, or segmentation.`;
    if (typeof m.loss_weight !== "number" || !Number.isFinite(m.loss_weight) || m.loss_weight <= 0)
      return `Measurement ${m.key}: loss_weight must be a positive number.`;

    if (m.type === "classification") {
      if (!Array.isArray(m.classes) || m.classes.length === 0)
        return `Measurement ${m.key}: at least one class is required.`;
      const classNamesList = m.classes.map((c) => c.name);
      if (new Set(classNamesList).size !== classNamesList.length)
        return `Measurement ${m.key}: class names must be unique.`;
      for (const c of m.classes) {
        if (!c.name || !TOKEN_RE.test(c.name))
          return `Measurement ${m.key}: class ${JSON.stringify(c.name)} must be a single alphanumeric token.`;
      }
      if (m.background_class != null && !classNamesList.includes(m.background_class))
        return `Measurement ${m.key}: background_class must be one of its own classes.`;
    } else if (m.type === "regression") {
      if (typeof m.min !== "number" || typeof m.max !== "number" || !(m.min < m.max))
        return `Measurement ${m.key}: min and max must be numbers with min < max.`;
    } else if (m.type === "segmentation") {
      if (!Array.isArray(m.seg_classes) || m.seg_classes.length < 2)
        return `Measurement ${m.key}: segmentation needs at least 2 mask classes (e.g. background + one subject).`;
      const segNames = m.seg_classes.map((c) => c.name);
      if (new Set(segNames).size !== segNames.length)
        return `Measurement ${m.key}: mask class names must be unique.`;
      for (const c of m.seg_classes) {
        if (!c.name || !TOKEN_RE.test(c.name))
          return `Measurement ${m.key}: mask class ${JSON.stringify(c.name)} must be a single alphanumeric token.`;
        if (!c.color || !COLOR_RE.test(c.color))
          return `Measurement ${m.key}: mask class ${c.name} needs a valid hex color.`;
      }
    }

    if (m.applies_when) {
      const { key, equals, not_equals } = m.applies_when;
      if (key === m.key) return `Measurement ${m.key}: applies_when cannot reference itself.`;
      const parent = t.measurements.find((p) => p.key === key);
      if (!parent) return `Measurement ${m.key}: applies_when references unknown measurement ${JSON.stringify(key)}.`;
      if (parent.type !== "classification")
        return `Measurement ${m.key}: applies_when must reference a classification measurement.`;
      if ((equals === undefined) === (not_equals === undefined))
        return `Measurement ${m.key}: applies_when must set exactly one of equals/not_equals.`;
      const targetValue = (equals ?? not_equals) as string;
      if (!(parent.classes ?? []).some((c) => c.name === targetValue))
        return `Measurement ${m.key}: applies_when value ${JSON.stringify(targetValue)} is not a class of ${parent.key}.`;
    }
  }

  if (!t.measurements.some((m) => m.type === "classification" && m.background_class))
    return "At least one classification measurement must declare a background_class (a \"no subject\" class), so the model has negatives to train against.";

  return null;
}
