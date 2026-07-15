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
// Species is just another classification measurement — one class per
// species — not a special schema-level concept with a single "active" one.
// Unlike the rest of DEFAULT_SCHEMA below, it isn't part of the starting
// set at all: an admin adds it themselves from the Structure editor once
// they know which species they're tracking, with no preset class baked in.
//
// This module holds the shared TypeScript shape, the DEFAULT_SCHEMA fallback
// (kept in sync with the SQL seed in
// supabase/migrations/20260716000010_drop_background_class_requirement.sql and
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

// A band of a regression measurement's predicted value, with its own preset
// explanation/recommendation copy — e.g. 0-30 "Low severity", 31-60
// "Moderate severity", 61-100 "Severe". Half-open on the low end: a value
// matches the first range with min <= value <= max, so ranges should be
// contiguous and non-overlapping (validated in validateSchema below).
export interface RangeDef {
  min: number;
  max: number;
  explanation?: string;
  recommendation?: string;
}

// One gating condition: a classification measurement (`key`) must equal (or
// not equal) a given class name. Exactly one of equals/not_equals should be
// set.
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
  // Gates a measurement so it's only meaningful (supervised in training,
  // shown in the UI) when EVERY condition in this list holds — e.g.
  // disease_severity applies_when [disease not_equals "NoDisease"]; a
  // measurement can list several conditions (all must be satisfied,
  // AND-combined) to depend on more than one sibling measurement at once.
  // Absent/empty means "always applies".
  applies_when?: AppliesWhen[] | null;
  // classification only
  background_class?: string | null; // name of the "no subject" class, if any
  classes?: ClassDef[];
  // regression only
  unit?: string;
  min?: number;
  max?: number;
  // Optional preset explanation/recommendation copy per band of the
  // predicted value (e.g. "0-30 -> Low severity"). No admin UI forces this
  // to cover the whole [min, max] span — a value outside every range simply
  // gets no preset copy.
  ranges?: RangeDef[];
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
const WHEN_SEAWEED_PRESENT: AppliesWhen[] = [{ key: "seaweed_presence", equals: "Yes" }];

// A 0–100 (or other-range) lab/quality regression. Keeps the long block below
// readable. Everything here is a starting point, not a fixed backbone — any
// of it (including seaweed_presence/species/health_status/disease/colour
// below) can be freely edited or removed from the admin Structure editor.
function labRegression(
  key: string,
  label: string,
  unit: string,
  max: number,
  applies_when: AppliesWhen[] = WHEN_SEAWEED_PRESENT
): MeasurementDef {
  return { key, label, type: "regression", loss_weight: 0.5, unit, min: 0, max, applies_when };
}

// Fallback used when no schema row exists yet. Mirrors the SQL seed.
export const DEFAULT_SCHEMA: SchemaDoc = {
  // Retained for schema compatibility; health status is now a labeled
  // classification (below), no longer derived from a numeric score.
  health_moderate_min: 45.0,
  health_healthy_min: 75.0,
  measurements: [
    // A plain classification: is there a seaweed specimen in the frame?
    {
      key: "seaweed_presence",
      label: "Seaweed presence",
      type: "classification",
      loss_weight: 1.0,
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
    // Species is just another classification, same as any other measurement
    // — but unlike the rest of this starting set, it has no default class:
    // which species you're tracking is entirely up to you. Add it yourself
    // from the Structure editor ("+ Add measurement") with one class per
    // species you actually collect.
    // The overall health label — an explicit class, not a bucketed score.
    {
      key: "health_status",
      label: "Health status",
      type: "classification",
      loss_weight: 1.0,
      applies_when: WHEN_SEAWEED_PRESENT,
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
    // Named diseases + an explicit "no disease" class. Add a class per
    // disease you want the model to recognise.
    {
      key: "disease",
      label: "Disease",
      type: "classification",
      loss_weight: 0.5,
      applies_when: WHEN_SEAWEED_PRESENT,
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
      applies_when: [{ key: "disease", not_equals: "NoDisease" }],
    },
    labRegression("dried", "Dried", "%", 100),
    labRegression("decayed", "Decayed", "%", 100),
    // Observed colour, a fixed palette (not free text).
    {
      key: "colour",
      label: "Colour",
      type: "classification",
      loss_weight: 0.5,
      applies_when: WHEN_SEAWEED_PRESENT,
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
    labRegression("carrageenan_yield", "Carrageenan Yield", "%", 100),
    labRegression("gel_strength", "Gel Strength", "g/cm²", 2000),
    labRegression("viscosity", "Viscosity", "cP", 1000),
    labRegression("daily_growth_rate", "Daily Growth Rate", "%/day", 100),
    labRegression("mineral_ca", "Mineral Content — Ca", "mg/kg", 100000),
    labRegression("mineral_mg", "Mineral Content — Mg", "mg/kg", 100000),
    labRegression("mineral_k", "Mineral Content — K", "mg/kg", 100000),
    labRegression("mineral_na", "Mineral Content — Na", "mg/kg", 100000),
    labRegression("caw", "Clean Anhydrous Weed (CAW)", "%", 100),
    labRegression("impurities", "Impurities", "%", 100),
    labRegression("sulfate_content", "Sulfate Content", "%", 100),
    labRegression("acid_insoluble_ash", "Acid-Insoluble Ash", "%", 100),
    labRegression("ash_content", "Ash Content", "%", 100),
  ],
};

// --- Derivation helpers ----------------------------------------------------

export function findMeasurement(doc: SchemaDoc, key: string): MeasurementDef | undefined {
  return doc.measurements.find((m) => m.key === key);
}

export function classificationMeasurements(doc: SchemaDoc): MeasurementDef[] {
  return doc.measurements.filter((m) => m.type === "classification");
}

// applies_when used to be a single condition object; it's now a list of
// AND-combined conditions. Normalizes a doc loaded from storage (or a
// hand-crafted payload) so a measurement whose applies_when is still the old
// single-object shape becomes a one-element list, and anything else
// (already a list, or null/absent) passes through unchanged.
export function normalizeSchemaDoc(doc: SchemaDoc): SchemaDoc {
  return {
    ...doc,
    measurements: doc.measurements.map((m) => {
      const aw = m.applies_when as unknown;
      if (aw && !Array.isArray(aw)) return { ...m, applies_when: [aw as AppliesWhen] };
      return m;
    }),
  };
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

// The first range (of a regression measurement) that `value` falls into, if
// any. Mirrors MeasurementDef.range_for in ml/config.py — keep both in sync.
export function rangeForValue(measurement: MeasurementDef, value: number): RangeDef | undefined {
  return (measurement.ranges ?? []).find((r) => value >= r.min && value <= r.max);
}

// Whether `measurement` is active given the current values of *other*
// measurements (keyed by measurement key -> chosen class name). A measurement
// with no applies_when (or an empty one) is always active; with several
// conditions, every one of them must hold (AND). Values are read as unknown
// so this works with both the client's Record<string,string> class-value
// state and the server's parsed-JSON measurements payload.
export function measurementApplies(measurement: MeasurementDef, values: Record<string, unknown>): boolean {
  const conditions = measurement.applies_when;
  if (!conditions || conditions.length === 0) return true;
  return conditions.every((cond) => {
    const parentValue = values[cond.key];
    if (parentValue === undefined || parentValue === null) return false;
    if (cond.equals !== undefined) return parentValue === cond.equals;
    if (cond.not_equals !== undefined) return parentValue !== cond.not_equals;
    return true;
  });
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
      if (m.ranges && m.ranges.length > 0) {
        const sorted = [...m.ranges].sort((a, b) => a.min - b.min);
        for (const r of sorted) {
          if (typeof r.min !== "number" || typeof r.max !== "number" || !(r.min < r.max))
            return `Measurement ${m.key}: each range needs numeric min < max.`;
          if (r.min < m.min || r.max > m.max)
            return `Measurement ${m.key}: range ${r.min}–${r.max} must fall within ${m.min}–${m.max}.`;
        }
        for (let i = 1; i < sorted.length; i++) {
          const prev = sorted[i - 1]!;
          const curr = sorted[i]!;
          if (curr.min < prev.max)
            return `Measurement ${m.key}: ranges ${prev.min}–${prev.max} and ${curr.min}–${curr.max} overlap.`;
        }
      }
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

    if (m.applies_when && m.applies_when.length > 0) {
      const seenKeys = new Set<string>();
      for (const { key, equals, not_equals } of m.applies_when) {
        if (key === m.key) return `Measurement ${m.key}: applies_when cannot reference itself.`;
        if (seenKeys.has(key)) return `Measurement ${m.key}: applies_when lists ${JSON.stringify(key)} more than once.`;
        seenKeys.add(key);
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
  }

  return null;
}
