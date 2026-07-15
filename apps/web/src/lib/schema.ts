// The measurement schema — the admin-editable definition of every per-image
// measurement the model predicts (classification, regression, or
// segmentation), plus preset explanation/recommendation copy per class. This
// used to be a fixed condition/severity/disease-subtype taxonomy hardcoded
// across ml/config.py, ml/src/inference/explanations.py, and this file; it is
// now a versioned JSONB document stored in Supabase (table
// measurement_schema), so an admin can add a whole new measurement (e.g.
// "moisture", "gel_strength", "biofouling") without any code change. New
// measurements with no ground-truth data yet simply stay masked (untrained)
// until values arrive.
//
// Species is just another classification measurement (see DEFAULT_SCHEMA
// below) — one class per species, admin-extensible like "disease" — not a
// special schema-level concept with a single "active" one.
//
// This module holds the shared TypeScript shape, the DEFAULT_SCHEMA fallback
// (kept in sync with the SQL seed in
// supabase/migrations/20260715000007_species_as_classification.sql and
// ml/config.py's fallback), plus validation/derivation helpers usable from
// both server and client code.

export type MeasurementType = "classification" | "regression" | "segmentation";

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
  // A "must-required" measurement that ships with the app. Locked measurements
  // can't be removed or have their key/type reconfigured from the admin
  // Structure editor — they're the fixed backbone every dataset collects
  // (seaweed presence, health status, the lab-quality metrics, …). Purely a
  // UI/authoring guard; the ML pipeline treats a locked measurement like any
  // other.
  locked?: boolean;
  // For a locked classification whose class list is still meant to grow (e.g.
  // "disease" — the admin adds a class per named disease). When false/absent
  // on a locked measurement, its classes are fixed too (e.g. colour's palette,
  // health status' Healthy/Moderate/Low).
  extensible_classes?: boolean;
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
  // Legacy display-level thresholds, relevant only to a schema whose health
  // measurement is still a regressed health_score (see
  // ml/src/inference/predictor.py's _derive_level fallback for a
  // pre-restructure checkpoint): at or above health_healthy_min ->
  // "Healthy"; at or above health_moderate_min (but below healthy) ->
  // "Moderate"; otherwise "Low". The current required schema assigns
  // health_status directly as a classification instead, so these are
  // optional and have no admin UI of their own.
  health_moderate_min?: number;
  health_healthy_min?: number;
  measurements: MeasurementDef[];
}

// "Seaweed present?" gates every subject-level measurement below: they only
// apply when a specimen is actually in frame.
const WHEN_SEAWEED_PRESENT: AppliesWhen = { key: "seaweed_presence", equals: "Yes" };

// A locked 0–100 (or other-range) lab/quality regression that ships as part of
// the required schema. Keeps the long block below readable.
function requiredRegression(
  key: string,
  label: string,
  unit: string,
  max: number,
  applies_when: AppliesWhen = WHEN_SEAWEED_PRESENT
): MeasurementDef {
  return { key, label, type: "regression", loss_weight: 0.5, unit, min: 0, max, applies_when, locked: true };
}

// Fallback used when no schema row exists yet. Mirrors the SQL seed.
export const DEFAULT_SCHEMA: SchemaDoc = {
  // Retained for schema compatibility; health status is now a labeled
  // classification (below), no longer derived from a numeric score.
  health_moderate_min: 45.0,
  health_healthy_min: 75.0,
  measurements: [
    // The primary classifier: is there a seaweed specimen in the frame at all?
    // "No" is the background / no-subject class the model trains against.
    {
      key: "seaweed_presence",
      label: "Seaweed presence",
      type: "classification",
      loss_weight: 1.0,
      background_class: "No",
      locked: true,
      classes: [
        {
          name: "Yes",
          explanation: "A seaweed specimen was detected in this image.",
          recommendation: "Continue with the assessment below.",
        },
        {
          name: "No",
          explanation: "No seaweed specimen was detected in this image.",
          recommendation: "Point the camera at a seaweed specimen, filling the frame, and try again.",
        },
      ],
    },
    // Species is just another classification: one class per species you
    // collect, admin-extensible (add a class per new species) — there's no
    // separate "active species" concept, and no cap of one species per
    // deployment.
    {
      key: "species",
      label: "Species",
      type: "classification",
      loss_weight: 1.0,
      applies_when: WHEN_SEAWEED_PRESENT,
      locked: true,
      extensible_classes: true,
      classes: [{ name: "Kappaphycus_alvarezii" }],
    },
    // The overall health label — an explicit class, not a bucketed score.
    {
      key: "health_status",
      label: "Health status",
      type: "classification",
      loss_weight: 1.0,
      applies_when: WHEN_SEAWEED_PRESENT,
      locked: true,
      classes: [
        {
          name: "Healthy",
          explanation: "Vivid, even coloration with intact branching and no whitening, lesions, or breakage.",
          recommendation: "Continue routine monitoring. No action needed.",
        },
        {
          name: "Moderate",
          explanation: "Some discoloration or minor structural loss, but the specimen is largely intact.",
          recommendation: "Increase monitoring frequency and check water quality (temperature, salinity).",
        },
        {
          name: "Low",
          explanation: "Extensive discoloration, tissue loss, or structural breakdown across the specimen.",
          recommendation: "Remove affected fragments to prevent spread and investigate the cause promptly.",
        },
      ],
    },
    // Named diseases + an explicit "no disease" class. Extensible: the admin
    // adds one class per disease they want the model to recognise.
    {
      key: "disease",
      label: "Disease",
      type: "classification",
      loss_weight: 0.5,
      applies_when: WHEN_SEAWEED_PRESENT,
      locked: true,
      extensible_classes: true,
      classes: [
        { name: "NoDisease", explanation: "No disease detected." },
        {
          name: "IceIce",
          note: "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity.",
        },
        { name: "Epiphyte", note: "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow." },
        { name: "Bacterial", note: "Possible bacterial infection: isolate and consult a specialist before any treatment." },
        { name: "Bleaching", note: "Bleaching suspected: check for temperature/light stress and relocate if possible." },
      ],
    },
    // Disease severity applies only once a disease (other than "no disease")
    // has been recorded.
    {
      key: "disease_severity",
      label: "Disease severity",
      type: "regression",
      loss_weight: 0.5,
      unit: "score",
      min: 0,
      max: 100,
      applies_when: { key: "disease", not_equals: "NoDisease" },
      locked: true,
    },
    requiredRegression("dried", "Dried", "%", 100),
    requiredRegression("decayed", "Decayed", "%", 100),
    // Observed colour, a fixed palette (not free text).
    {
      key: "colour",
      label: "Colour",
      type: "classification",
      loss_weight: 0.5,
      applies_when: WHEN_SEAWEED_PRESENT,
      locked: true,
      classes: [
        { name: "Green" },
        { name: "Red" },
        { name: "Brown" },
        { name: "Yellow" },
        { name: "Orange" },
        { name: "White" },
        { name: "Black" },
      ],
    },
    // Lab / quality-assay metrics.
    requiredRegression("carrageenan_yield", "Carrageenan Yield", "%", 100),
    requiredRegression("gel_strength", "Gel Strength", "g/cm²", 2000),
    requiredRegression("viscosity", "Viscosity", "cP", 1000),
    requiredRegression("daily_growth_rate", "Daily Growth Rate", "%/day", 100),
    requiredRegression("mineral_ca", "Mineral Content — Ca", "mg/kg", 100000),
    requiredRegression("mineral_mg", "Mineral Content — Mg", "mg/kg", 100000),
    requiredRegression("mineral_k", "Mineral Content — K", "mg/kg", 100000),
    requiredRegression("mineral_na", "Mineral Content — Na", "mg/kg", 100000),
    requiredRegression("caw", "Clean Anhydrous Weed (CAW)", "%", 100),
    requiredRegression("impurities", "Impurities", "%", 100),
    requiredRegression("sulfate_content", "Sulfate Content", "%", 100),
    requiredRegression("acid_insoluble_ash", "Acid-Insoluble Ash", "%", 100),
    requiredRegression("ash_content", "Ash Content", "%", 100),
  ],
};

// The keys of every locked (must-required) measurement, derived from
// DEFAULT_SCHEMA so there's a single source of truth. Used by the admin
// editor (to lock controls) and validation (to reject removing them).
export const REQUIRED_MEASUREMENT_KEYS: readonly string[] = DEFAULT_SCHEMA.measurements
  .filter((m) => m.locked)
  .map((m) => m.key);

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

// --- Slug/key derivation -----------------------------------------------------
// Measurement keys are stable identifiers threaded through the DB, the JSON
// schema, and the Python/ML pipeline — not something an admin should
// hand-type. The editor derives them from the human-readable label instead;
// exported so it (and validation) agree on the exact same derivation.

export function slugifyKey(label: string): string {
  const slug = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return slug || "measurement";
}

// --- Validation ------------------------------------------------------------

const KEY_RE = /^[a-z][a-z0-9_]*$/; // measurement keys are manifest/JSON identifiers
// Class/seg-class names stay folder-name safe; underscores are allowed (e.g.
// species classes like "Kappaphycus_alvarezii").
const TOKEN_RE = /^[A-Za-z0-9_]+$/;
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

  // Optional legacy thresholds — only validated if present at all, since the
  // required schema no longer has any UI to set them (health_status is
  // assigned directly as a classification now).
  if (t.health_moderate_min !== undefined && !isFraction(t.health_moderate_min))
    return "health_moderate_min must be a number between 0 and 100.";
  if (t.health_healthy_min !== undefined && !isFraction(t.health_healthy_min))
    return "health_healthy_min must be a number between 0 and 100.";
  if (
    t.health_moderate_min !== undefined &&
    t.health_healthy_min !== undefined &&
    t.health_healthy_min <= t.health_moderate_min
  )
    return "health_healthy_min must be greater than health_moderate_min.";

  if (!Array.isArray(t.measurements) || t.measurements.length === 0)
    return "At least one measurement is required.";

  const keys = t.measurements.map((m) => m.key);
  if (new Set(keys).size !== keys.length) return "Measurement keys must be unique.";

  for (const m of t.measurements) {
    if (!m.key || !KEY_RE.test(m.key))
      return `Measurement key ${JSON.stringify(m.key)} must be lowercase snake_case (e.g. "health_score").`;
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
          return `Measurement ${m.key}: class ${JSON.stringify(c.name)} must be a single alphanumeric/underscore token.`;
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

  // Locked, must-required measurements can't be dropped or retyped (the admin
  // editor enforces this in the UI; this guards the API against a hand-edited
  // payload that removes them).
  for (const req of DEFAULT_SCHEMA.measurements.filter((m) => m.locked)) {
    const found = t.measurements.find((m) => m.key === req.key);
    if (!found) return `Required measurement ${JSON.stringify(req.key)} (${req.label}) cannot be removed.`;
    if (found.type !== req.type)
      return `Required measurement ${JSON.stringify(req.key)} must stay a ${req.type}.`;
  }

  return null;
}
