"use client";

import { useEffect, useState } from "react";
import {
  DEFAULT_SCHEMA,
  getPrimaryClassification,
  measurementApplies,
  type SchemaDoc,
} from "@/lib/schema";

const INPUT = "rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900";
const LABEL = "flex flex-col gap-1 text-sm font-medium text-slate-700";

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

  const [showMore, setShowMore] = useState(false);
  const [farm, setFarm] = useState("");
  const [camera, setCamera] = useState("");
  const [capturedAt, setCapturedAt] = useState("");
  const [waterTemperatureC, setWaterTemperatureC] = useState("");
  const [salinityPpt, setSalinityPpt] = useState("");
  const [depthM, setDepthM] = useState("");
  const [gpsLat, setGpsLat] = useState("");
  const [gpsLng, setGpsLng] = useState("");

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
    setFarm("");
    setCamera("");
    setCapturedAt("");
    setWaterTemperatureC("");
    setSalinityPpt("");
    setDepthM("");
    setGpsLat("");
    setGpsLng("");
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
    formData.append("farm", farm);
    formData.append("camera", camera);
    formData.append("capturedAt", capturedAt);
    formData.append("waterTemperatureC", waterTemperatureC);
    formData.append("salinityPpt", salinityPpt);
    formData.append("depthM", depthM);
    formData.append("gpsLat", gpsLat);
    formData.append("gpsLng", gpsLng);

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
    <form onSubmit={handleSubmit} className="mv-card flex flex-col gap-4 p-6">
      <h2 className="text-lg font-semibold text-slate-900">Label a new photo</h2>

      <label className={LABEL}>
        Photo
        <input
          type="file"
          accept="image/*"
          required
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          className="text-sm"
        />
      </label>

      {/* One control per schema measurement, in schema order. A measurement
          with an applies_when that isn't satisfied by the current selections
          (e.g. disease_subtype when condition != Disease) is hidden. */}
      {schema.measurements.map((m) => {
        if (!measurementApplies(m, classValues)) return null;

        if (m.type === "classification") {
          return (
            <label key={m.key} className={LABEL}>
              {m.label}
              <select
                required
                value={classValues[m.key] ?? ""}
                onChange={(event) => setClassValues((prev) => ({ ...prev, [m.key]: event.target.value }))}
                className={INPUT}
              >
                {(m.classes ?? []).map((c) => (
                  <option key={c.name} value={c.name}>
                    {m.background_class === c.name ? `${c.name} (no subject)` : c.name}
                  </option>
                ))}
              </select>
            </label>
          );
        }

        if (m.type === "regression") {
          return (
            <label key={m.key} className="flex flex-col gap-1 text-sm text-slate-700">
              {m.label} ({m.min ?? 0}–{m.max ?? 100}{m.unit ? ` ${m.unit}` : ""}, optional)
              <input
                type="number"
                min={m.min ?? 0}
                max={m.max ?? 100}
                value={numberValues[m.key] ?? ""}
                onChange={(event) => setNumberValues((prev) => ({ ...prev, [m.key]: event.target.value }))}
                placeholder="anchor if blank"
                className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
              />
            </label>
          );
        }

        // segmentation
        return (
          <label key={m.key} className="flex flex-col gap-1 text-sm text-slate-700">
            {m.label} mask (optional)
            <input
              type="file"
              accept="image/*"
              onChange={(event) =>
                setMaskFiles((prev) => ({ ...prev, [m.key]: event.target.files?.[0] ?? null }))
              }
              className="text-sm"
            />
          </label>
        );
      })}

      {!isBackground && (
        <div className="grid gap-4 sm:grid-cols-2">
          <label className={LABEL}>
            Species
            <input
              type="text"
              value={species}
              onChange={(event) => setSpecies(event.target.value)}
              className={INPUT}
            />
          </label>
          <label className={LABEL}>
            Colour
            <input
              type="text"
              value={colour}
              onChange={(event) => setColour(event.target.value)}
              placeholder="e.g. dark green"
              className={INPUT}
            />
          </label>
        </div>
      )}

      <label className={LABEL}>
        Notes
        <textarea
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          rows={2}
          className={INPUT}
        />
      </label>

      <button
        type="button"
        onClick={() => setShowMore((prev) => !prev)}
        className="self-start text-sm font-semibold text-ocean-700"
      >
        {showMore ? "Hide" : "Show"} more metadata
      </button>

      {showMore && (
        <div className="grid gap-4 sm:grid-cols-3">
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Farm
            <input
              type="text"
              value={farm}
              onChange={(event) => setFarm(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Camera
            <input
              type="text"
              value={camera}
              onChange={(event) => setCamera(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Captured at
            <input
              type="datetime-local"
              value={capturedAt}
              onChange={(event) => setCapturedAt(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            GPS latitude
            <input
              type="number"
              step="any"
              value={gpsLat}
              onChange={(event) => setGpsLat(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            GPS longitude
            <input
              type="number"
              step="any"
              value={gpsLng}
              onChange={(event) => setGpsLng(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Water temp (°C)
            <input
              type="number"
              step="any"
              value={waterTemperatureC}
              onChange={(event) => setWaterTemperatureC(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Salinity (ppt)
            <input
              type="number"
              step="any"
              value={salinityPpt}
              onChange={(event) => setSalinityPpt(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            Depth (m)
            <input
              type="number"
              step="any"
              value={depthM}
              onChange={(event) => setDepthM(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
            />
          </label>
        </div>
      )}

      {error && <p className="text-sm text-coral-600">{error}</p>}

      <button type="submit" disabled={submitting} className="mv-btn-blue self-start">
        {submitting ? "Uploading…" : "Add to dataset"}
      </button>
    </form>
  );
}
