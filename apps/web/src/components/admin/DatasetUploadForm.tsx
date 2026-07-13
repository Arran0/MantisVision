"use client";

import { useState } from "react";
import { CONDITIONS, SEVERITIES, DISEASE_SUBTYPES } from "@/lib/taxonomy";

export function DatasetUploadForm({ onUploaded }: { onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [condition, setCondition] = useState<string>("Healthy");
  const [severity, setSeverity] = useState<string>(SEVERITIES[0]);
  const [subtype, setSubtype] = useState<string>(DISEASE_SUBTYPES[0]);
  const [diseaseName, setDiseaseName] = useState("");
  const [species, setSpecies] = useState("Kappaphycus alvarezii");
  const [colour, setColour] = useState("");
  const [notes, setNotes] = useState("");

  // Optional numeric overrides — blank falls back to the heuristic anchors in ml/config.py.
  const [healthScore, setHealthScore] = useState("");
  const [driedPct, setDriedPct] = useState("");
  const [decayedPct, setDecayedPct] = useState("");

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

  const isDisease = condition === "Disease";
  const isBackground = condition === "Background";

  function resetForm() {
    setFile(null);
    setColour("");
    setNotes("");
    setDiseaseName("");
    setHealthScore("");
    setDriedPct("");
    setDecayedPct("");
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

    const formData = new FormData();
    formData.append("file", file);
    formData.append("condition", condition);
    if (isDisease) {
      formData.append("severity", severity);
      formData.append("subtype", subtype);
      formData.append("diseaseName", diseaseName);
    }
    if (!isBackground) {
      formData.append("species", species);
      formData.append("colour", colour);
      formData.append("healthScore", healthScore);
      formData.append("driedPct", driedPct);
      formData.append("decayedPct", decayedPct);
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

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Photo
        <input
          type="file"
          accept="image/*"
          required
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          className="text-sm"
        />
      </label>

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Condition
        <select
          required
          value={condition}
          onChange={(event) => setCondition(event.target.value)}
          className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
        >
          {CONDITIONS.map((option) => (
            <option key={option} value={option}>
              {option === "Background" ? "Background (no seaweed)" : option}
            </option>
          ))}
        </select>
      </label>

      {isDisease && (
        <div className="grid gap-4 sm:grid-cols-3">
          <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
            Severity
            <select
              value={severity}
              onChange={(event) => setSeverity(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
            >
              {SEVERITIES.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
            Subtype
            <select
              value={subtype}
              onChange={(event) => setSubtype(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
            >
              {DISEASE_SUBTYPES.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
            Disease name (optional)
            <input
              type="text"
              value={diseaseName}
              onChange={(event) => setDiseaseName(event.target.value)}
              placeholder="e.g. Vibrio_sp"
              className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
            />
          </label>
        </div>
      )}

      {!isBackground && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
              Species
              <input
                type="text"
                value={species}
                onChange={(event) => setSpecies(event.target.value)}
                className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
              Colour
              <input
                type="text"
                value={colour}
                onChange={(event) => setColour(event.target.value)}
                placeholder="e.g. dark green"
                className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
              />
            </label>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <label className="flex flex-col gap-1 text-sm text-slate-700">
              Health score (0–100, optional)
              <input
                type="number"
                min="0"
                max="100"
                value={healthScore}
                onChange={(event) => setHealthScore(event.target.value)}
                placeholder="anchor if blank"
                className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm text-slate-700">
              Dried % (optional)
              <input
                type="number"
                min="0"
                max="100"
                value={driedPct}
                onChange={(event) => setDriedPct(event.target.value)}
                placeholder="anchor if blank"
                className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
              />
            </label>
            <label className="flex flex-col gap-1 text-sm text-slate-700">
              Decayed % (optional)
              <input
                type="number"
                min="0"
                max="100"
                value={decayedPct}
                onChange={(event) => setDecayedPct(event.target.value)}
                placeholder="anchor if blank"
                className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-sm text-slate-900"
              />
            </label>
          </div>
        </>
      )}

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Notes
        <textarea
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          rows={2}
          className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
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
