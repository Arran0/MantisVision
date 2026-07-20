"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  DEFAULT_SCHEMA,
  classificationMeasurements,
  slugifyKey,
  validateSchema,
  type AppliesWhen,
  type ClassDef,
  type MeasurementDef,
  type MeasurementType,
  type RangeDef,
  type SegClassDef,
  type SchemaDoc,
} from "@/lib/schema";
import {
  AdminButton,
  AdminCard,
  AdminField,
  AdminInput,
  AdminSelect,
  AdminTextarea,
  sectionHeadingClass,
} from "@/components/admin/ui";

function defaultsForType(type: MeasurementType): Partial<MeasurementDef> {
  if (type === "classification") {
    return {
      classes: [{ name: "" }],
      background_class: null,
      unit: undefined,
      min: undefined,
      max: undefined,
      seg_classes: undefined,
    };
  }
  if (type === "regression") {
    return { classes: undefined, background_class: undefined, unit: "", min: 0, max: 100, ranges: [], seg_classes: undefined };
  }
  return {
    classes: undefined,
    background_class: undefined,
    unit: undefined,
    min: undefined,
    max: undefined,
    ranges: undefined,
    seg_classes: [
      { name: "background", color: "#000000" },
      { name: "subject", color: "#22c55e" },
    ],
  };
}

function emptyMeasurement(): MeasurementDef {
  return { key: "measurement", label: "", type: "classification", loss_weight: 1.0, ...defaultsForType("classification") };
}

const fade = {
  initial: { opacity: 0, height: 0 },
  animate: { opacity: 1, height: "auto" },
  exit: { opacity: 0, height: 0 },
  transition: { duration: 0.18 },
};

export function MeasurementSchemaEditor() {
  const [doc, setDoc] = useState<SchemaDoc>(DEFAULT_SCHEMA);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const response = await fetch("/api/member/schema");
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

  function patchMeasurementLabel(index: number, label: string) {
    patchMeasurement(index, { label, key: slugifyKey(label) });
  }

  function patchClass(measurementIndex: number, classIndex: number, next: Partial<ClassDef>) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) => {
        if (i !== measurementIndex) return m;
        const oldName = m.classes?.[classIndex]?.name;
        const classes = (m.classes ?? []).map((c, j) => (j === classIndex ? { ...c, ...next } : c));
        // Keep background_class pointing at the same class if it's the one
        // being renamed here, so a rename doesn't silently invalidate it.
        const background_class =
          next.name !== undefined && m.background_class === oldName ? next.name : m.background_class;
        return { ...m, classes, background_class };
      }),
    }));
  }

  // Removing a class that's currently the background_class would otherwise
  // leave a dangling reference — clear it along with the class.
  function removeClass(measurementIndex: number, classIndex: number) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) => {
        if (i !== measurementIndex) return m;
        const removedName = m.classes?.[classIndex]?.name;
        const classes = (m.classes ?? []).filter((_, j) => j !== classIndex);
        const background_class = m.background_class === removedName ? null : m.background_class;
        return { ...m, classes, background_class };
      }),
    }));
  }

  function addRange(measurementIndex: number) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex ? { ...m, ranges: [...(m.ranges ?? []), { min: m.min ?? 0, max: m.max ?? 100 }] } : m
      ),
    }));
  }

  function patchRange(measurementIndex: number, rangeIndex: number, next: Partial<RangeDef>) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex
          ? { ...m, ranges: (m.ranges ?? []).map((r, j) => (j === rangeIndex ? { ...r, ...next } : r)) }
          : m
      ),
    }));
  }

  function removeRange(measurementIndex: number, rangeIndex: number) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex ? { ...m, ranges: (m.ranges ?? []).filter((_, j) => j !== rangeIndex) } : m
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

  // applies_when is a list of AND-combined conditions — a measurement can
  // depend on several sibling classifications at once (e.g. "only when
  // seaweed_presence == Yes AND disease != NoDisease").
  function addAppliesWhen(measurementIndex: number, firstClassification: MeasurementDef | undefined) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) => {
        if (i !== measurementIndex || !firstClassification) return m;
        const cond: AppliesWhen = { key: firstClassification.key, equals: firstClassification.classes?.[0]?.name ?? "" };
        return { ...m, applies_when: [...(m.applies_when ?? []), cond] };
      }),
    }));
  }

  function removeAppliesWhen(measurementIndex: number, condIndex: number) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex ? { ...m, applies_when: (m.applies_when ?? []).filter((_, j) => j !== condIndex) } : m
      ),
    }));
  }

  function patchAppliesWhen(measurementIndex: number, condIndex: number, next: AppliesWhen) {
    setDoc((prev) => ({
      ...prev,
      measurements: prev.measurements.map((m, i) =>
        i === measurementIndex
          ? { ...m, applies_when: (m.applies_when ?? []).map((c, j) => (j === condIndex ? next : c)) }
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
      const response = await fetch("/api/member/schema", {
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

  if (loading) return <p className="text-sm text-zinc-500">Loading schema…</p>;

  const classifications = classificationMeasurements(doc);

  return (
    <div className="flex flex-col gap-5">
      {/* Measurements ---------------------------------------------------- */}
      <div className="flex flex-col gap-3">
        <div>
          <h2 className={`${sectionHeadingClass} mt-0`}>Measurements</h2>
          <p className="text-sm text-zinc-600">
            One measurement = one column per photo and one thing the model predicts (classification, regression, or
            segmentation). Everything below is editable — add one to teach the model something new; it stays
            untrained until labeled photos supply values.
          </p>
        </div>
        <AnimatePresence initial={false}>
          {doc.measurements.map((m, i) => {
            const appliesWhen = m.applies_when ?? [];
            const availableParents = classifications.filter(
              (c) => c.key !== m.key && !appliesWhen.some((cond) => cond.key === c.key)
            );
            return (
              <motion.div key={i} {...fade}>
                <AdminCard className="flex flex-col gap-3 p-5">
                  <div className="flex flex-wrap items-end gap-3">
                    <AdminField label="Label" className="flex-1 min-w-[10rem]">
                      <AdminInput value={m.label} onChange={(e) => patchMeasurementLabel(i, e.target.value)} placeholder="e.g. Gel strength" />
                    </AdminField>
                    <AdminField label="Type" className="min-w-[9rem]">
                      <AdminSelect
                        value={m.type}
                        onChange={(e) => {
                          const type = e.target.value as MeasurementType;
                          patchMeasurement(i, { type, ...defaultsForType(type) });
                        }}
                      >
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                        <option value="segmentation">Segmentation</option>
                      </AdminSelect>
                    </AdminField>
                    <AdminField label="Loss weight" className="w-24">
                      <AdminInput
                        type="number"
                        min={0}
                        step="0.1"
                        value={m.loss_weight}
                        onChange={(e) => patchMeasurement(i, { loss_weight: Number(e.target.value) })}
                      />
                    </AdminField>
                    <AdminButton
                      type="button"
                      variant="ghost"
                      className="mb-0.5 text-rose-600 hover:bg-rose-50"
                      onClick={() => patch({ measurements: doc.measurements.filter((_, j) => j !== i) })}
                    >
                      Remove
                    </AdminButton>
                  </div>
                  <p className="-mt-1 text-xs text-zinc-400">
                    Internal key: <code className="rounded bg-zinc-100 px-1 py-0.5">{m.key}</code> — follows the label;
                    renaming an existing measurement changes this, so already-collected values under the old key won&rsquo;t
                    carry over automatically.
                  </p>

                  {/* applies_when — a repeatable list of AND-combined conditions -- */}
                  <div className="flex flex-col gap-2 rounded-sm bg-zinc-50 p-3">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-dewberry-600">
                      Only applies when {appliesWhen.length > 1 ? "(all must hold)" : ""}
                    </span>
                    {appliesWhen.map((cond, ci) => {
                      const parent = classifications.find((c) => c.key === cond.key);
                      return (
                        <div key={ci} className="flex flex-wrap items-end gap-3">
                          <AdminField label="Measurement" className="min-w-[10rem]">
                            <AdminSelect
                              value={cond.key}
                              onChange={(e) => {
                                const key = e.target.value;
                                const target = classifications.find((c) => c.key === key);
                                patchAppliesWhen(i, ci, { key, equals: target?.classes?.[0]?.name ?? "" });
                              }}
                            >
                              {/* Include the condition's current parent even if it's
                                  otherwise filtered out by availableParents (already
                                  used by a sibling condition), so the select shows it. */}
                              {[parent, ...availableParents]
                                .filter((c): c is MeasurementDef => !!c)
                                .filter((c, idx, arr) => arr.findIndex((x) => x.key === c.key) === idx)
                                .map((c) => (
                                  <option key={c.key} value={c.key}>
                                    {c.label}
                                  </option>
                                ))}
                            </AdminSelect>
                          </AdminField>
                          <AdminField label="Comparison" className="min-w-[8rem]">
                            <AdminSelect
                              value={cond.equals !== undefined ? "equals" : "not_equals"}
                              onChange={(e) => {
                                const val = cond.equals ?? cond.not_equals ?? "";
                                patchAppliesWhen(
                                  i,
                                  ci,
                                  e.target.value === "equals" ? { key: cond.key, equals: val } : { key: cond.key, not_equals: val }
                                );
                              }}
                            >
                              <option value="equals">equals</option>
                              <option value="not_equals">does not equal</option>
                            </AdminSelect>
                          </AdminField>
                          <AdminField label="Value" className="min-w-[8rem]">
                            <AdminSelect
                              value={cond.equals ?? cond.not_equals ?? ""}
                              onChange={(e) => {
                                const isEquals = cond.equals !== undefined;
                                patchAppliesWhen(
                                  i,
                                  ci,
                                  isEquals ? { key: cond.key, equals: e.target.value } : { key: cond.key, not_equals: e.target.value }
                                );
                              }}
                            >
                              {(parent?.classes ?? []).map((c) => (
                                <option key={c.name} value={c.name}>
                                  {c.name}
                                </option>
                              ))}
                            </AdminSelect>
                          </AdminField>
                          <AdminButton
                            type="button"
                            variant="ghost"
                            className="mb-0.5 text-rose-600 hover:bg-rose-50"
                            onClick={() => removeAppliesWhen(i, ci)}
                          >
                            Remove condition
                          </AdminButton>
                        </div>
                      );
                    })}
                    {availableParents.length > 0 && (
                      <AdminButton
                        type="button"
                        variant="ghost"
                        className="self-start"
                        onClick={() => addAppliesWhen(i, availableParents[0])}
                      >
                        + Add condition
                      </AdminButton>
                    )}
                    {appliesWhen.length === 0 && (
                      <p className="text-xs text-zinc-400">Always applies — no conditions set.</p>
                    )}
                  </div>

                  {/* Type-specific editor ------------------------------------ */}
                  <AnimatePresence mode="wait" initial={false}>
                    <motion.div key={m.type} {...fade} className="flex flex-col gap-3">
                      {m.type === "classification" &&
                        (m.classes ?? []).map((c, ci) => (
                          <div key={ci} className="flex flex-col gap-2 rounded-sm border border-zinc-200 p-3">
                            <div className="flex items-end gap-3">
                              <AdminField label="Class name" className="flex-1 min-w-[8rem]">
                                <AdminInput value={c.name} onChange={(e) => patchClass(i, ci, { name: e.target.value })} />
                              </AdminField>
                              <AdminButton
                                type="button"
                                variant="ghost"
                                className="mb-0.5 text-rose-600 hover:bg-rose-50"
                                onClick={() => removeClass(i, ci)}
                              >
                                Remove
                              </AdminButton>
                            </div>
                            <AdminField label="Explanation (shown to end users)">
                              <AdminTextarea
                                rows={2}
                                value={c.explanation ?? ""}
                                onChange={(e) => patchClass(i, ci, { explanation: e.target.value })}
                              />
                            </AdminField>
                            <AdminField label="Recommendation">
                              <AdminTextarea
                                rows={2}
                                value={c.recommendation ?? ""}
                                onChange={(e) => patchClass(i, ci, { recommendation: e.target.value })}
                              />
                            </AdminField>
                            <AdminField label="Note (appended to a parent measurement's recommendation, if any)">
                              <AdminInput value={c.note ?? ""} onChange={(e) => patchClass(i, ci, { note: e.target.value })} />
                            </AdminField>
                          </div>
                        ))}
                      {m.type === "classification" && (
                        <AdminButton
                          type="button"
                          variant="ghost"
                          className="self-start"
                          onClick={() => patchMeasurement(i, { classes: [...(m.classes ?? []), { name: "" }] })}
                        >
                          + Add class
                        </AdminButton>
                      )}

                      {m.type === "regression" && (
                        <div className="flex flex-col gap-3">
                          <div className="grid gap-3 sm:grid-cols-3">
                            <AdminField label="Unit">
                              <AdminInput value={m.unit ?? ""} onChange={(e) => patchMeasurement(i, { unit: e.target.value })} />
                            </AdminField>
                            <AdminField label="Min">
                              <AdminInput
                                type="number"
                                value={m.min ?? 0}
                                onChange={(e) => patchMeasurement(i, { min: Number(e.target.value) })}
                              />
                            </AdminField>
                            <AdminField label="Max">
                              <AdminInput
                                type="number"
                                value={m.max ?? 100}
                                onChange={(e) => patchMeasurement(i, { max: Number(e.target.value) })}
                              />
                            </AdminField>
                          </div>

                          <div className="flex flex-col gap-2 rounded-sm bg-zinc-50 p-3">
                            <span className="text-[10px] font-bold uppercase tracking-widest text-dewberry-600">
                              Explanation / recommendation by range (optional)
                            </span>
                            <p className="-mt-1 text-xs text-zinc-500">
                              Give different preset copy for different bands of the predicted value — e.g. 0–30 vs.
                              31–60 vs. 61–100. A value outside every range below just gets no preset copy.
                            </p>
                            {(m.ranges ?? []).map((r, ri) => (
                              <div key={ri} className="flex flex-col gap-2 rounded-sm border border-zinc-200 p-3">
                                <div className="flex items-end gap-3">
                                  <AdminField label="From" className="w-24">
                                    <AdminInput
                                      type="number"
                                      value={r.min}
                                      onChange={(e) => patchRange(i, ri, { min: Number(e.target.value) })}
                                    />
                                  </AdminField>
                                  <AdminField label="To" className="w-24">
                                    <AdminInput
                                      type="number"
                                      value={r.max}
                                      onChange={(e) => patchRange(i, ri, { max: Number(e.target.value) })}
                                    />
                                  </AdminField>
                                  <AdminButton
                                    type="button"
                                    variant="ghost"
                                    className="mb-0.5 text-rose-600 hover:bg-rose-50"
                                    onClick={() => removeRange(i, ri)}
                                  >
                                    Remove
                                  </AdminButton>
                                </div>
                                <AdminField label="Explanation (shown to end users)">
                                  <AdminTextarea
                                    rows={2}
                                    value={r.explanation ?? ""}
                                    onChange={(e) => patchRange(i, ri, { explanation: e.target.value })}
                                  />
                                </AdminField>
                                <AdminField label="Recommendation">
                                  <AdminTextarea
                                    rows={2}
                                    value={r.recommendation ?? ""}
                                    onChange={(e) => patchRange(i, ri, { recommendation: e.target.value })}
                                  />
                                </AdminField>
                              </div>
                            ))}
                            <AdminButton type="button" variant="ghost" className="self-start" onClick={() => addRange(i)}>
                              + Add range
                            </AdminButton>
                          </div>
                        </div>
                      )}

                      {m.type === "segmentation" && (
                        <div className="flex flex-col gap-3">
                          {(m.seg_classes ?? []).map((c, si) => (
                            <div key={si} className="flex items-end gap-3">
                              <AdminField label="Mask class name" className="flex-1 min-w-[8rem]">
                                <AdminInput value={c.name} onChange={(e) => patchSegClass(i, si, { name: e.target.value })} />
                              </AdminField>
                              <AdminField label="Color" className="w-20">
                                <input
                                  type="color"
                                  className="h-[2.375rem] w-full rounded-sm border border-zinc-300"
                                  value={c.color}
                                  onChange={(e) => patchSegClass(i, si, { color: e.target.value })}
                                />
                              </AdminField>
                              <AdminButton
                                type="button"
                                variant="ghost"
                                className="mb-0.5 text-rose-600 hover:bg-rose-50"
                                onClick={() =>
                                  patchMeasurement(i, { seg_classes: (m.seg_classes ?? []).filter((_, j) => j !== si) })
                                }
                              >
                                Remove
                              </AdminButton>
                            </div>
                          ))}
                          <AdminButton
                            type="button"
                            variant="ghost"
                            className="self-start"
                            onClick={() =>
                              patchMeasurement(i, {
                                seg_classes: [...(m.seg_classes ?? []), { name: "", color: "#888888" }],
                              })
                            }
                          >
                            + Add mask class
                          </AdminButton>
                        </div>
                      )}
                    </motion.div>
                  </AnimatePresence>
                </AdminCard>
              </motion.div>
            );
          })}
        </AnimatePresence>
        <AdminButton type="button" variant="ghost" className="self-start" onClick={() => patch({ measurements: [...doc.measurements, emptyMeasurement()] })}>
          + Add measurement
        </AdminButton>
      </div>

      {error && <p className="text-sm text-rose-600">{error}</p>}
      {message && <p className="text-sm text-seaweed-600">{message}</p>}

      <div className="flex items-center gap-3">
        <AdminButton type="button" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save schema"}
        </AdminButton>
        <span className="text-xs text-zinc-500">Existing labeled images and trained models are unaffected until you retrain.</span>
      </div>
    </div>
  );
}
