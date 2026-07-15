"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  DEFAULT_SCHEMA,
  classificationMeasurements,
  slugifyKey,
  validateSchema,
  type ClassDef,
  type MeasurementDef,
  type MeasurementType,
  type SegClassDef,
  type SchemaDoc,
} from "@/lib/schema";
import {
  AdminBadge,
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
    return { classes: undefined, background_class: undefined, unit: "", min: 0, max: 100, seg_classes: undefined };
  }
  return {
    classes: undefined,
    background_class: undefined,
    unit: undefined,
    min: undefined,
    max: undefined,
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

  function patchMeasurementLabel(index: number, label: string) {
    patchMeasurement(index, { label, key: slugifyKey(label) });
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

  if (loading) return <p className="text-sm text-zinc-500">Loading schema…</p>;

  const classifications = classificationMeasurements(doc);

  return (
    <div className="flex flex-col gap-5">
      {/* Measurements ---------------------------------------------------- */}
      <div className="flex flex-col gap-3">
        <div>
          <h2 className={`${sectionHeadingClass} mt-0`}>Measurements</h2>
          <p className="text-sm text-zinc-600">
            Each measurement is one column collected per photo and one head the model predicts: a classification
            (named classes), a regression (a continuous value), or a segmentation (a per-pixel mask). The
            <span className="mx-1"><AdminBadge tone="ocean">Required</AdminBadge></span>
            ones are the fixed backbone — you can&rsquo;t remove or retype them, though <em>Species</em> and
            <em> Disease</em> stay extensible: add one class per species or disease you want the model to recognise,
            no need to mark any single one &ldquo;active&rdquo;. Add your own measurements below to teach the model a
            new metric; it stays untrained until labeled photos supply values for it.
          </p>
        </div>
        <AnimatePresence initial={false}>
          {doc.measurements.map((m, i) => {
            const locked = !!m.locked;
            // A locked classification whose class list is still meant to grow
            // (e.g. Disease) keeps its class editors live; every other locked
            // measurement is fully read-only.
            const classesEditable = !locked || !!m.extensible_classes;
            return (
            <motion.div key={i} {...fade}>
              <AdminCard className="flex flex-col gap-3 p-5">
                <div className="flex flex-wrap items-end gap-3">
                  <AdminField label="Label" className="flex-1 min-w-[10rem]">
                    <AdminInput
                      value={m.label}
                      disabled={locked}
                      onChange={(e) => patchMeasurementLabel(i, e.target.value)}
                      placeholder="e.g. Gel strength"
                    />
                  </AdminField>
                  <AdminField label="Type" className="min-w-[9rem]">
                    <AdminSelect
                      value={m.type}
                      disabled={locked}
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
                      disabled={locked}
                      onChange={(e) => patchMeasurement(i, { loss_weight: Number(e.target.value) })}
                    />
                  </AdminField>
                  {locked ? (
                    <div className="mb-2.5">
                      <AdminBadge tone="ocean">Required</AdminBadge>
                    </div>
                  ) : (
                    <AdminButton
                      type="button"
                      variant="ghost"
                      className="mb-0.5 text-rose-600 hover:bg-rose-50"
                      onClick={() => patch({ measurements: doc.measurements.filter((_, j) => j !== i) })}
                    >
                      Remove
                    </AdminButton>
                  )}
                </div>
                <p className="-mt-1 text-xs text-zinc-400">
                  Internal key: <code className="rounded bg-zinc-100 px-1 py-0.5">{m.key}</code>
                  {locked
                    ? " — fixed for this required measurement."
                    : " — follows the label; renaming an existing measurement changes this, so already-collected values under the old key won’t carry over automatically."}
                </p>

                {/* applies_when ------------------------------------------- */}
                <div className="flex flex-wrap items-end gap-3 rounded-lg bg-zinc-50 p-3">
                  <AdminField label="Only applies when" className="min-w-[10rem]">
                    <AdminSelect
                      value={m.applies_when?.key ?? ""}
                      disabled={locked}
                      onChange={(e) => {
                        const key = e.target.value;
                        if (!key) return patchMeasurement(i, { applies_when: null });
                        patchMeasurement(i, {
                          applies_when: {
                            key,
                            equals: classifications.find((c) => c.key === key)?.classes?.[0]?.name ?? "",
                          },
                        });
                      }}
                    >
                      <option value="">— always —</option>
                      {classifications
                        .filter((c) => c.key !== m.key)
                        .map((c) => (
                          <option key={c.key} value={c.key}>
                            {c.label}
                          </option>
                        ))}
                    </AdminSelect>
                  </AdminField>
                  {m.applies_when &&
                    (() => {
                      const parent = classifications.find((c) => c.key === m.applies_when!.key);
                      return (
                        <>
                          <AdminField label="Comparison" className="min-w-[8rem]">
                            <AdminSelect
                              value={m.applies_when!.equals !== undefined ? "equals" : "not_equals"}
                              disabled={locked}
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
                            </AdminSelect>
                          </AdminField>
                          <AdminField label="Value" className="min-w-[8rem]">
                            <AdminSelect
                              value={m.applies_when!.equals ?? m.applies_when!.not_equals ?? ""}
                              disabled={locked}
                              onChange={(e) => {
                                const isEquals = m.applies_when!.equals !== undefined;
                                patchMeasurement(i, {
                                  applies_when: isEquals
                                    ? { key: m.applies_when!.key, equals: e.target.value }
                                    : { key: m.applies_when!.key, not_equals: e.target.value },
                                });
                              }}
                            >
                              {(parent?.classes ?? []).map((c) => (
                                <option key={c.name} value={c.name}>
                                  {c.name}
                                </option>
                              ))}
                            </AdminSelect>
                          </AdminField>
                        </>
                      );
                    })()}
                </div>

                {/* Type-specific editor ------------------------------------ */}
                <AnimatePresence mode="wait" initial={false}>
                  <motion.div key={m.type} {...fade} className="flex flex-col gap-3">
                    {m.type === "classification" &&
                      (m.classes ?? []).map((c, ci) => (
                        <div key={ci} className="flex flex-col gap-2 rounded-lg border border-zinc-200 p-3">
                          <div className="flex items-end gap-3">
                            <AdminField label="Class name" className="flex-1 min-w-[8rem]">
                              <AdminInput
                                value={c.name}
                                disabled={!classesEditable}
                                onChange={(e) => patchClass(i, ci, { name: e.target.value })}
                              />
                            </AdminField>
                            {classesEditable && (
                              <AdminButton
                                type="button"
                                variant="ghost"
                                className="mb-0.5 text-rose-600 hover:bg-rose-50"
                                disabled={(m.classes ?? []).length <= 1}
                                onClick={() => patchMeasurement(i, { classes: (m.classes ?? []).filter((_, j) => j !== ci) })}
                              >
                                Remove
                              </AdminButton>
                            )}
                          </div>
                          <AdminField label="Explanation (shown to end users)">
                            <AdminTextarea
                              rows={2}
                              value={c.explanation ?? ""}
                              disabled={!classesEditable}
                              onChange={(e) => patchClass(i, ci, { explanation: e.target.value })}
                            />
                          </AdminField>
                          <AdminField label="Recommendation">
                            <AdminTextarea
                              rows={2}
                              value={c.recommendation ?? ""}
                              disabled={!classesEditable}
                              onChange={(e) => patchClass(i, ci, { recommendation: e.target.value })}
                            />
                          </AdminField>
                          <AdminField label="Note (appended to a parent measurement's recommendation, if any)">
                            <AdminInput
                              value={c.note ?? ""}
                              disabled={!classesEditable}
                              onChange={(e) => patchClass(i, ci, { note: e.target.value })}
                            />
                          </AdminField>
                        </div>
                      ))}
                    {m.type === "classification" && classesEditable && (
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
                      <div className="grid gap-3 sm:grid-cols-3">
                        <AdminField label="Unit">
                          <AdminInput
                            value={m.unit ?? ""}
                            disabled={locked}
                            onChange={(e) => patchMeasurement(i, { unit: e.target.value })}
                          />
                        </AdminField>
                        <AdminField label="Min">
                          <AdminInput
                            type="number"
                            value={m.min ?? 0}
                            disabled={locked}
                            onChange={(e) => patchMeasurement(i, { min: Number(e.target.value) })}
                          />
                        </AdminField>
                        <AdminField label="Max">
                          <AdminInput
                            type="number"
                            value={m.max ?? 100}
                            disabled={locked}
                            onChange={(e) => patchMeasurement(i, { max: Number(e.target.value) })}
                          />
                        </AdminField>
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
                                className="h-[2.375rem] w-full rounded-lg border border-zinc-300"
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
