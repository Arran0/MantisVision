"use client";

import { useEffect, useState } from "react";
import {
  DEFAULT_TAXONOMY,
  validateTaxonomy,
  type ConditionDef,
  type SubtypeDef,
  type TaxonomyDoc,
} from "@/lib/taxonomy";

const INPUT =
  "rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900";
const LABEL = "flex flex-col gap-1 text-sm font-medium text-slate-700";

function emptyCondition(name: string): ConditionDef {
  return {
    name,
    is_background: false,
    fixed_severity: null,
    requires_subtype: false,
    health_score_anchor: 50,
    health_score_anchors_by_severity: {},
    dried_extent_anchor: 0,
    decayed_extent_anchor: 0,
    explanation: "",
    recommendation: "",
  };
}

export function TaxonomyEditor() {
  const [doc, setDoc] = useState<TaxonomyDoc>(DEFAULT_TAXONOMY);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const response = await fetch("/api/admin/taxonomy");
      const payload = await response.json().catch(() => null);
      if (response.ok && payload?.taxonomy) setDoc(payload.taxonomy as TaxonomyDoc);
      setLoading(false);
    })();
  }, []);

  // Immutable helpers ------------------------------------------------------
  function patch(next: Partial<TaxonomyDoc>) {
    setDoc((prev) => ({ ...prev, ...next }));
  }
  function patchCondition(index: number, next: Partial<ConditionDef>) {
    setDoc((prev) => ({
      ...prev,
      conditions: prev.conditions.map((c, i) => (i === index ? { ...c, ...next } : c)),
    }));
  }
  function patchSubtype(index: number, next: Partial<SubtypeDef>) {
    setDoc((prev) => ({
      ...prev,
      disease_subtypes: prev.disease_subtypes.map((s, i) => (i === index ? { ...s, ...next } : s)),
    }));
  }

  async function save() {
    setError(null);
    setMessage(null);
    const problem = validateTaxonomy(doc);
    if (problem) {
      setError(problem);
      return;
    }
    setSaving(true);
    try {
      const response = await fetch("/api/admin/taxonomy", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taxonomy: doc }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? "Failed to save.");
      setMessage("Saved. New labeling and validation use this immediately; the model picks it up on the next retrain + promote.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <p className="text-sm text-slate-500">Loading taxonomy…</p>;

  return (
    <div className="flex flex-col gap-6">
      {/* Species -------------------------------------------------------- */}
      <section className="mv-card flex flex-col gap-3 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Species</h2>
        <p className="text-sm text-slate-600">
          The active species prefixes every dataset folder. Slugs must be folder-safe (letters, numbers,
          underscores).
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

      {/* Severities + disease threshold -------------------------------- */}
      <section className="mv-card flex flex-col gap-3 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Severities</h2>
        <label className={LABEL}>
          Comma-separated severity tokens (e.g. Moderate, Low)
          <input
            className={INPUT}
            value={doc.severities.join(", ")}
            onChange={(e) =>
              patch({
                severities: e.target.value
                  .split(",")
                  .map((x) => x.trim())
                  .filter(Boolean),
              })
            }
          />
        </label>
        <label className={LABEL}>
          Disease “Moderate” threshold (a subtype prediction scoring at or above this shows as Moderate, else
          Low)
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

      {/* Conditions ---------------------------------------------------- */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Conditions (buckets)</h2>
        <p className="text-sm text-slate-600">
          Each condition is a training class. Anchors are the fallback ground-truth numbers used when an
          uploaded image has no explicit value. The explanation/recommendation is the preset copy shown to end
          users for that prediction.
        </p>
        {doc.conditions.map((c, i) => (
          <div key={i} className="mv-card flex flex-col gap-3 p-5">
            <div className="flex flex-wrap items-end gap-3">
              <label className={LABEL}>
                Name (folder-safe token)
                <input className={INPUT} value={c.name} onChange={(e) => patchCondition(i, { name: e.target.value })} />
              </label>
              <label className="flex items-center gap-2 pb-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  checked={c.is_background}
                  onChange={(e) => patchCondition(i, { is_background: e.target.checked })}
                />
                Background (no-seaweed class)
              </label>
              <label className="flex items-center gap-2 pb-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  checked={c.requires_subtype}
                  onChange={(e) => patchCondition(i, { requires_subtype: e.target.checked })}
                />
                Carries a subtype + per-image severity
              </label>
              <label className={LABEL}>
                Fixed severity
                <select
                  className={INPUT}
                  value={c.fixed_severity ?? ""}
                  onChange={(e) => patchCondition(i, { fixed_severity: e.target.value || null })}
                >
                  <option value="">— none —</option>
                  {doc.severities.map((sev) => (
                    <option key={sev} value={sev}>
                      {sev}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="pb-2 text-sm font-semibold text-coral-600"
                onClick={() => patch({ conditions: doc.conditions.filter((_, j) => j !== i) })}
              >
                Remove
              </button>
            </div>

            {!c.is_background && (
              <div className="grid gap-3 sm:grid-cols-3">
                {c.requires_subtype ? (
                  doc.severities.map((sev) => (
                    <label key={sev} className={LABEL}>
                      Health anchor ({sev})
                      <input
                        type="number"
                        min={0}
                        max={100}
                        className={INPUT}
                        value={c.health_score_anchors_by_severity[sev] ?? ""}
                        onChange={(e) =>
                          patchCondition(i, {
                            health_score_anchors_by_severity: {
                              ...c.health_score_anchors_by_severity,
                              [sev]: Number(e.target.value),
                            },
                          })
                        }
                      />
                    </label>
                  ))
                ) : (
                  <label className={LABEL}>
                    Health score anchor
                    <input
                      type="number"
                      min={0}
                      max={100}
                      className={INPUT}
                      value={c.health_score_anchor ?? ""}
                      onChange={(e) =>
                        patchCondition(i, {
                          health_score_anchor: e.target.value === "" ? null : Number(e.target.value),
                        })
                      }
                    />
                  </label>
                )}
                <label className={LABEL}>
                  Dried extent anchor
                  <input
                    type="number"
                    min={0}
                    max={100}
                    className={INPUT}
                    value={c.dried_extent_anchor}
                    onChange={(e) => patchCondition(i, { dried_extent_anchor: Number(e.target.value) })}
                  />
                </label>
                <label className={LABEL}>
                  Decayed extent anchor
                  <input
                    type="number"
                    min={0}
                    max={100}
                    className={INPUT}
                    value={c.decayed_extent_anchor}
                    onChange={(e) => patchCondition(i, { decayed_extent_anchor: Number(e.target.value) })}
                  />
                </label>
              </div>
            )}

            <label className={LABEL}>
              Explanation (shown to end users)
              <textarea
                rows={2}
                className={INPUT}
                value={c.explanation}
                onChange={(e) => patchCondition(i, { explanation: e.target.value })}
              />
            </label>
            <label className={LABEL}>
              Recommendation
              <textarea
                rows={2}
                className={INPUT}
                value={c.recommendation}
                onChange={(e) => patchCondition(i, { recommendation: e.target.value })}
              />
            </label>
          </div>
        ))}
        <button
          type="button"
          className="self-start text-sm font-semibold text-ocean-700"
          onClick={() => patch({ conditions: [...doc.conditions, emptyCondition("")] })}
        >
          + Add condition
        </button>
      </section>

      {/* Disease subtypes --------------------------------------------- */}
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-slate-900">Disease subtypes</h2>
        {doc.disease_subtypes.map((s, i) => (
          <div key={i} className="mv-card flex flex-col gap-3 p-5 sm:flex-row sm:items-end">
            <label className={`${LABEL} sm:w-48`}>
              Name (token)
              <input className={INPUT} value={s.name} onChange={(e) => patchSubtype(i, { name: e.target.value })} />
            </label>
            <label className={`${LABEL} flex-1`}>
              Note (appended to Disease recommendation)
              <input className={INPUT} value={s.note} onChange={(e) => patchSubtype(i, { note: e.target.value })} />
            </label>
            <button
              type="button"
              className="pb-2 text-sm font-semibold text-coral-600"
              onClick={() => patch({ disease_subtypes: doc.disease_subtypes.filter((_, j) => j !== i) })}
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          className="self-start text-sm font-semibold text-ocean-700"
          onClick={() => patch({ disease_subtypes: [...doc.disease_subtypes, { name: "", note: "" }] })}
        >
          + Add subtype
        </button>
      </section>

      {error && <p className="text-sm text-coral-600">{error}</p>}
      {message && <p className="text-sm text-seaweed-600">{message}</p>}

      <div className="flex items-center gap-3">
        <button type="button" onClick={save} disabled={saving} className="mv-btn-blue self-start">
          {saving ? "Saving…" : "Save taxonomy"}
        </button>
        <span className="text-xs text-slate-500">
          Existing labeled images and trained models are unaffected until you retrain.
        </span>
      </div>
    </div>
  );
}
