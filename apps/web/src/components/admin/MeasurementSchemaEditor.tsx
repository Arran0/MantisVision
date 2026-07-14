"use client";

import { useEffect, useState } from "react";
import {
  DEFAULT_SCHEMA,
  validateSchema,
  type ClassDef,
  type MeasurementDef,
  type MeasurementType,
  type SegClassDef,
  type SchemaDoc,
} from "@/lib/schema";

const INPUT = "rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900";
const LABEL = "flex flex-col gap-1 text-sm font-medium text-slate-700";

function defaultsForType(type: MeasurementType): Partial<MeasurementDef> {
  if (type === "classification") {
    return { classes: [{ name: "" }], background_class: null, unit: undefined, min: undefined, max: undefined, seg_classes: undefined };
  }
  if (type === "regression") {
    return { classes: undefined, background_class: undefined, unit: "", min: 0, max: 100, seg_classes: undefined };
  }
  return { classes: undefined, background_class: undefined, unit: undefined, min: undefined, max: undefined, seg_classes: [{ name: "background", color: "#000000" }, { name: "subject", color: "#22c55e" }] };
}

function emptyMeasurement(): MeasurementDef {
  return { key: "", label: "", type: "classification", loss_weight: 1.0, ...defaultsForType("classification") };
}

export function MeasurementSchemaEditor() {
  const [doc, setDoc] = useState<SchemaDoc>(DEFAULT_SCHEMA);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const response = await fetch("/api/admin/schema");
      const payload = await response.json().catch(() => null);
      if (response.ok && payload?.schema) setDoc(payload.schema as SchemaDoc);
      setLoading(false);
    })();
  }, []);

  function patch(next: Partial<SchemaDoc>) {
    setDoc((prev) => ({ ...prev, ...next }));
  }
  function patchMeasurement(index: number, next: Partial<MeasurementDef>) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) => (i === index ? { ...m, ...next } : m)),
    }));
  }
  function patchClass(measurementIndex: number, classIndex: number, next: Partial<ClassDef>) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex
          ? { ...m, classes: (m.classes ?? []).map((c, j) => (j === classIndex ? { ...c, ...next } : c)) }
          : m
      ),
    }));
  }
  function patchSegClass(measurementIndex: number, segIndex: number, next: Partial<SegClassDef>) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex
          ? { ...m, seg_classes: (m.seg_classes ?? []).map((c, j) => (j === segIndex ? { ...c, ...next } : c)) }
          : m
      ),
    }));
  }

  async function save() {
    setError(null);
    setMessage(null);
    const problem = validateSchema(doc);
    if (problem) {
      setError(problem);
      return;
    }
    setSaving(true);
    try {
      const response = await fetch("/api/admin/schema", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ schema: doc }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? "Failed to save.");
      setMessage(
        "Saved. New labeling and validation use this immediately; the model picks it up on the next retrain + promote."
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <p className="text-sm text-slate-500">Loading schema…</p>;

  const classificationMeasurements = doc.measurements.filter((m) => m.type === "classification");

  return (
    <div className="flex flex-col gap-6">
      {/* Species -------------------------------------------------------- */}
      <section className="mv-card flex flex-col gap-3 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Species</h2>
        <p className="text-sm text-slate-600">
          The active species is the default recorded on new (non-background) labeled photos. Slugs must be
          folder-safe (letters, numbers, underscores).
        </p>
        {doc.species.map((s, i) => (
          <div key={i} className="flex flex-wrap items-end gap-3">
            <label className={LABEL}>
              Name
              <input
                className={INPUT}
                value={s.name}
                onChange={(e) =>
                  patch({ species: doc.species.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)) })
                }
              />
            </label>
            <label className={LABEL}>
              Slug
              <input
                className={INPUT}
                value={s.slug}
                onChange={(e) =>
                  patch({ species: doc.species.map((x, j) => (j === i ? { ...x, slug: e.target.value } : x)) })
                }
              />
            </label>
            <label className="flex items-center gap-2 pb-2 text-sm text-slate-700">
              <input
                type="radio"
                name="active-species"
                checked={doc.active_species_slug === s.slug}
                onChange={() => patch({ active_species_slug: s.slug })}
              />
              Active
            </label>
            <button
              type="button"
              className="pb-2 text-sm font-semibold text-coral-600"
              onClick={() => patch({ species: doc.species.filter((_, j) => j !== i) })}
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          className="self-start text-sm font-semibold text-ocean-700"
          onClick={() => patch({ species: [...doc.species, { name: "", slug: "" }] })}
        >
          + Add species
        </button>
      </section>

      {/* Disease threshold ---------------------------------------------- */}
      <section className="mv-card flex flex-col gap-3 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Display threshold</h2>
        <label className={LABEL}>
          Disease &ldquo;Moderate&rdquo; threshold (health score at or above this shows as Moderate, else Low)
          <input
            type="number"
            min={0}
            max={100}
            className={`${INPUT} max-w-32`}
            value={doc.disease_moderate_min}
            onChange={(e) => patch({ disease_moderate_min: Number(e.target.value) })}
          />
        </label>
      </section>

      {/* Measurements ---------------------------------------------------- */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Measurements</h2>
        <p className="text-sm text-slate-600">
          Each measurement is one head the model predicts: a classification (a set of named classes), a
          regression (a continuous 0–100-style value), or a segmentation (a per-pixel mask). Add a new one to
          teach the model a new quality metric — e.g. moisture, gel strength, biofouling — with no code change;
          it stays untrained until labeled photos supply values for it.
        </p>
        {doc.measurements.map((m, i) => {
          const otherClassifications = classificationMeasurements.filter((c) => c.key !== m.key);
          const appliesToParent = m.applies_when ? classificationMeasurements.find((c) => c.key === m.applies_when!.key) : undefined;
          return (
            <div key={i} className="mv-card flex flex-col gap-3 p-5">
              <div className="flex flex-wrap items-end gap-3">
                <label className={LABEL}>
                  Key (snake_case)
                  <input
                    className={INPUT}
                    value={m.key}
                    onChange={(e) => patchMeasurement(i, { key: e.target.value })}
                    placeholder="e.g. gel_strength"
                  />
                </label>
                <label className={LABEL}>
                  Label
                  <input className={INPUT} value={m.label} onChange={(e) => patchMeasurement(i, { label: e.target.value })} />
                </label>
                <label className={LABEL}>
                  Type
                  <select
                    className={INPUT}
                    value={m.type}
                    onChange={(e) => {
                      const type = e.target.value as MeasurementType;
                      patchMeasurement(i, { type, ...defaultsForType(type) });
                    }}
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                    <option value="segmentation">Segmentation</option>
                  </select>
                </label>
                <label className={LABEL}>
                  Loss weight
                  <input
                    type="number"
                    min={0}
                    step="0.1"
                    className={`${INPUT} max-w-24`}
                    value={m.loss_weight}
                    onChange={(e) => patchMeasurement(i, { loss_weight: Number(e.target.value) })}
                  />
                </label>
                <button
                  type="button"
                  className="pb-2 text-sm font-semibold text-coral-600"
                  onClick={() => patch({ measurements: doc.measurements.filter((_, j) => j !== i) })}
                >
                  Remove
                </button>
              </div>

              {/* applies_when ------------------------------------------- */}
              <div className="flex flex-wrap items-end gap-3 rounded-lg bg-slate-50 p-3">
                <label className={LABEL}>
                  Only applies when
                  <select
                    className={INPUT}
                    value={m.applies_when?.key ?? ""}
                    onChange={(e) => {
                      const key = e.target.value;
                      if (!key) return patchMeasurement(i, { applies_when: null });
                      patchMeasurement(i, { applies_when: { key, equals: classificationMeasurements.find((c) => c.key === key)?.classes?.[0]?.name ?? "" } });
                    }}
                  >
                    <option value="">— always —</option>
                    {otherClassifications.map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                </label>
                {m.applies_when && (
                  <>
                    <label className={LABEL}>
                      Comparison
                      <select
                        className={INPUT}
                        value={m.applies_when.equals !== undefined ? "equals" : "not_equals"}
                        onChange={(e) => {
                          const val = m.applies_when!.equals ?? m.applies_when!.not_equals ?? "";
                          patchMeasurement(i, {
                            applies_when:
                              e.target.value === "equals"
                                ? { key: m.applies_when!.key, equals: val }
                                : { key: m.applies_when!.key, not_equals: val },
                          });
                        }}
                      >
                        <option value="equals">equals</option>
                        <option value="not_equals">does not equal</option>
                      </select>
                    </label>
                    <label className={LABEL}>
                      Value
                      <select
                        className={INPUT}
                        value={m.applies_when.equals ?? m.applies_when.not_equals ?? ""}
                        onChange={(e) => {
                          const isEquals = m.applies_when!.equals !== undefined;
                          patchMeasurement(i, {
                            applies_when: isEquals
                              ? { key: m.applies_when!.key, equals: e.target.value }
                              : { key: m.applies_when!.key, not_equals: e.target.value },
                          });
                        }}
                      >
                        {(appliesToParent?.classes ?? []).map((c) => (
                          <option key={c.name} value={c.name}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
              </div>

              {/* Type-specific editor ------------------------------------ */}
              {m.type === "classification" && (
                <div className="flex flex-col gap-3">
                  <label className={LABEL}>
                    Background class (no-subject class, if any)
                    <select
                      className={`${INPUT} max-w-64`}
                      value={m.background_class ?? ""}
                      onChange={(e) => patchMeasurement(i, { background_class: e.target.value || null })}
                    >
                      <option value="">— none —</option>
                      {(m.classes ?? []).map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  {(m.classes ?? []).map((c, ci) => (
                    <div key={ci} className="flex flex-col gap-2 rounded-lg border border-slate-200 p-3">
                      <div className="flex items-end gap-3">
                        <label className={LABEL}>
                          Class name (token)
                          <input className={INPUT} value={c.name} onChange={(e) => patchClass(i, ci, { name: e.target.value })} />
                        </label>
                        <button
                          type="button"
                          className="pb-2 text-sm font-semibold text-coral-600"
                          onClick={() => patchMeasurement(i, { classes: (m.classes ?? []).filter((_, j) => j !== ci) })}
                        >
                          Remove class
                        </button>
                      </div>
                      <label className={LABEL}>
                        Explanation (shown to end users)
                        <textarea
                          rows={2}
                          className={INPUT}
                          value={c.explanation ?? ""}
                          onChange={(e) => patchClass(i, ci, { explanation: e.target.value })}
                        />
                      </label>
                      <label className={LABEL}>
                        Recommendation
                        <textarea
                          rows={2}
                          className={INPUT}
                          value={c.recommendation ?? ""}
                          onChange={(e) => patchClass(i, ci, { recommendation: e.target.value })}
                        />
                      </label>
                      <label className={LABEL}>
                        Note (appended to the parent measurement&apos;s recommendation, if any)
                        <input className={INPUT} value={c.note ?? ""} onChange={(e) => patchClass(i, ci, { note: e.target.value })} />
                      </label>
                    </div>
                  ))}
                  <button
                    type="button"
                    className="self-start text-sm font-semibold text-ocean-700"
                    onClick={() => patchMeasurement(i, { classes: [...(m.classes ?? []), { name: "" }] })}
                  >
                    + Add class
                  </button>
                </div>
              )}

              {m.type === "regression" && (
                <div className="grid gap-3 sm:grid-cols-3">
                  <label className={LABEL}>
                    Unit
                    <input className={INPUT} value={m.unit ?? ""} onChange={(e) => patchMeasurement(i, { unit: e.target.value })} />
                  </label>
                  <label className={LABEL}>
                    Min
                    <input
                      type="number"
                      className={INPUT}
                      value={m.min ?? 0}
                      onChange={(e) => patchMeasurement(i, { min: Number(e.target.value) })}
                    />
                  </label>
                  <label className={LABEL}>
                    Max
                    <input
                      type="number"
                      className={INPUT}
                      value={m.max ?? 100}
                      onChange={(e) => patchMeasurement(i, { max: Number(e.target.value) })}
                    />
                  </label>
                </div>
              )}

              {m.type === "segmentation" && (
                <div className="flex flex-col gap-3">
                  {(m.seg_classes ?? []).map((c, si) => (
                    <div key={si} className="flex items-end gap-3">
                      <label className={LABEL}>
                        Mask class name (token)
                        <input className={INPUT} value={c.name} onChange={(e) => patchSegClass(i, si, { name: e.target.value })} />
                      </label>
                      <label className={LABEL}>
                        Color
                        <input
                          type="color"
                          className="h-10 w-16 rounded-lg border border-slate-300"
                          value={c.color}
                          onChange={(e) => patchSegClass(i, si, { color: e.target.value })}
                        />
                      </label>
                      <button
                        type="button"
                        className="pb-2 text-sm font-semibold text-coral-600"
                        onClick={() => patchMeasurement(i, { seg_classes: (m.seg_classes ?? []).filter((_, j) => j !== si) })}
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    className="self-start text-sm font-semibold text-ocean-700"
                    onClick={() =>
                      patchMeasurement(i, { seg_classes: [...(m.seg_classes ?? []), { name: "", color: "#888888" }] })
                    }
                  >
                    + Add mask class
                  </button>
                </div>
              )}
            </div>
          );
        })}
        <button
          type="button"
          className="self-start text-sm font-semibold text-ocean-700"
          onClick={() => patch({ measurements: [...doc.measurements, emptyMeasurement()] })}
        >
          + Add measurement
        </button>
      </section>

      {error && <p className="text-sm text-coral-600">{error}</p>}
      {message && <p className="text-sm text-seaweed-600">{message}</p>}

      <div className="flex items-center gap-3">
        <button type="button" onClick={save} disabled={saving} className="mv-btn-blue self-start">
          {saving ? "Saving…" : "Save schema"}
        </button>
        <span className="text-xs text-slate-500">
          Existing labeled images and trained models are unaffected until you retrain.
        </span>
      </div>
    </div>
  );
}
