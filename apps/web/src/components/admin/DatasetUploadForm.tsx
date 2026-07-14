"use client";

import { useEffect, useState } from "react";
import {
  DEFAULT_SCHEMA,
  getPrimaryClassification,
  measurementApplies,
  type SchemaDoc,
} from "@/lib/schema";
import { AdminButton, AdminCard, AdminField, AdminInput, AdminSelect, AdminTextarea, sectionHeadingClass } from "@/components/admin/ui";

export function DatasetUploadForm({ onUploaded }: { onUploaded: () => void }) {
  // Starts from the built-in defaults and swaps to the live, admin-edited
  // schema once it loads, so new measurements/species/classes appear here
  // with no code change.
  const [schema, setSchema] = useState<SchemaDoc>(DEFAULT_SCHEMA);
  const [file, setFile] = useState<File | null>(null);
  const [species, setSpecies] = useState("");

  // One value per measurement, keyed by measurement key. Classification ->
  // selected class name; regression -> raw number-input string (blank =
  // unset, falls back to the ML pipeline's anchor); segmentation -> an
  // uploaded mask file.
  const [classValues, setClassValues] = useState<Record<string, string>>({});
  const [numberValues, setNumberValues] = useState<Record<string, string>>({});
  const [maskFiles, setMaskFiles] = useState<Record<string, File | null>>({});

  const [colour, setColour] = useState("");
  const [notes, setNotes] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function applyDefaults(doc: SchemaDoc) {
    setSchema(doc);
    setClassValues((prev) => {
      const next = { ...prev };
      for (const m of doc.measurements) {
        if (m.type === "classification" && (!next[m.key] || !m.classes?.some((c) => c.name === next[m.key]))) {
          next[m.key] = m.classes?.[0]?.name ?? "";
        }
      }
      return next;
    });
    setSpecies((prev) => prev || doc.species.find((s) => s.slug === doc.active_species_slug)?.name || "");
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const response = await fetch("/api/admin/schema");
      const payload = await response.json().catch(() => null);
      if (!cancelled && response.ok && payload?.schema) applyDefaults(payload.schema as SchemaDoc);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const primary = getPrimaryClassification(schema);
  const isBackground = !!primary && classValues[primary.key] === primary.background_class;

  function resetForm() {
    setFile(null);
    setColour("");
    setNotes("");
    setNumberValues({});
    setMaskFiles({});
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Choose a photo first.");
      return;
    }
    setSubmitting(true);
    setError(null);

    const measurements: Record<string, string | number> = {};
    for (const m of schema.measurements) {
      if (!measurementApplies(m, classValues)) continue;
      if (m.type === "classification") {
        const value = classValues[m.key];
        if (value) measurements[m.key] = value;
      } else if (m.type === "regression") {
        const raw = numberValues[m.key];
        if (raw !== undefined && raw.trim() !== "") {
          const num = Number(raw);
          if (Number.isFinite(num)) measurements[m.key] = num;
        }
      }
      // segmentation values are attached server-side from the uploaded mask file
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("measurements", JSON.stringify(measurements));
    for (const m of schema.measurements) {
      if (m.type !== "segmentation" || !measurementApplies(m, classValues)) continue;
      const maskFile = maskFiles[m.key];
      if (maskFile) formData.append(`mask:${m.key}`, maskFile);
    }
    if (!isBackground) {
      formData.append("species", species);
      formData.append("colour", colour);
    }
    formData.append("notes", notes);

    try {
      const response = await fetch("/api/admin/dataset", { method: "POST", body: formData });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(payload?.error ?? `Upload failed (HTTP ${response.status}).`);
      }
      resetForm();
      onUploaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AdminCard className="p-5">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <h2 className={sectionHeadingClass}>Label a new photo</h2>

        <AdminField label="Photo">
          <input
            type="file"
            accept="image/*"
            required
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="block text-sm text-zinc-700"
          />
        </AdminField>

        {/* One control per schema measurement, in schema order. A measurement
            with an applies_when that isn't satisfied by the current selections
            (e.g. disease_subtype when condition != Disease) is hidden. */}
        {schema.measurements.map((m) => {
          if (!measurementApplies(m, classValues)) return null;

          if (m.type === "classification") {
            return (
              <AdminField key={m.key} label={m.label}>
                <AdminSelect
                  required
                  value={classValues[m.key] ?? ""}
                  onChange={(event) => setClassValues((prev) => ({ ...prev, [m.key]: event.target.value }))}
                >
                  {(m.classes ?? []).map((c) => (
                    <option key={c.name} value={c.name}>
                      {m.background_class === c.name ? `${c.name} (no subject)` : c.name}
                    </option>
                  ))}
                </AdminSelect>
              </AdminField>
            );
          }

          if (m.type === "regression") {
            return (
              <AdminField key={m.key} label={`${m.label} (${m.min ?? 0}–${m.max ?? 100}${m.unit ? ` ${m.unit}` : ""}, optional)`}>
                <AdminInput
                  type="number"
                  min={m.min ?? 0}
                  max={m.max ?? 100}
                  value={numberValues[m.key] ?? ""}
                  onChange={(event) => setNumberValues((prev) => ({ ...prev, [m.key]: event.target.value }))}
                  placeholder="anchor if blank"
                />
              </AdminField>
            );
          }

          // segmentation
          return (
            <AdminField key={m.key} label={`${m.label} mask (optional)`}>
              <input
                type="file"
                accept="image/*"
                onChange={(event) => setMaskFiles((prev) => ({ ...prev, [m.key]: event.target.files?.[0] ?? null }))}
                className="block text-sm text-zinc-700"
              />
            </AdminField>
          );
        })}

        {!isBackground && (
          <div className="grid gap-4 sm:grid-cols-2">
            <AdminField label="Species">
              <AdminInput type="text" value={species} onChange={(event) => setSpecies(event.target.value)} />
            </AdminField>
            <AdminField label="Colour">
              <AdminInput
                type="text"
                value={colour}
                onChange={(event) => setColour(event.target.value)}
                placeholder="e.g. dark green"
              />
            </AdminField>
          </div>
        )}

        <AdminField label="Notes">
          <AdminTextarea value={notes} onChange={(event) => setNotes(event.target.value)} rows={2} />
        </AdminField>

        {error && <p className="text-sm text-rose-600">{error}</p>}

        <AdminButton type="submit" disabled={submitting} className="self-start">
          {submitting ? "Uploading…" : "Add to dataset"}
        </AdminButton>
      </form>
    </AdminCard>
  );
}
