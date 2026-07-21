"use client";

import { useEffect, useRef, useState } from "react";
import { DEFAULT_SCHEMA, measurementApplies, type SchemaDoc } from "@/lib/schema";
import { AdminButton, AdminCard, AdminField, AdminInput, AdminSelect, AdminTextarea, sectionHeadingClass } from "@/components/admin/ui";

// The model trains at 224x224 regardless of source resolution (ml/config.py),
// so a phone photo's native 10-20MB / 4000px+ original buys nothing for
// classification/regression labels — it just makes the upload slow. Anything
// above this long edge gets downscaled and re-encoded client-side before
// upload; anything already small is sent untouched.
const MAX_UPLOAD_DIMENSION = 1600;
const UPLOAD_JPEG_QUALITY = 0.85;
// Skip re-encoding small files entirely — nothing to gain.
const COMPRESS_THRESHOLD_BYTES = 900 * 1024;

// Downscales+re-encodes an oversized photo on the client so upload time
// tracks the compressed size, not the camera's native resolution. Falls back
// to the original file untouched on any decode/encode failure (e.g. an
// unsupported format) or when it's already small enough.
async function prepareForUpload(file: File): Promise<File> {
  if (file.size <= COMPRESS_THRESHOLD_BYTES) return file;

  const bitmap = await createImageBitmap(file).catch(() => null);
  if (!bitmap) return file;

  const scale = Math.min(1, MAX_UPLOAD_DIMENSION / Math.max(bitmap.width, bitmap.height));
  if (scale >= 1) {
    bitmap.close();
    return file;
  }

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    return file;
  }
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", UPLOAD_JPEG_QUALITY));
  if (!blob) return file;

  const nameWithoutExt = file.name.replace(/\.[^./]+$/, "");
  return new File([blob], `${nameWithoutExt}.jpg`, { type: "image/jpeg" });
}

export function DatasetUploadForm({ onUploaded }: { onUploaded: () => void }) {
  // Starts from the built-in defaults and swaps to the live, admin-edited
  // schema once it loads, so new measurements/classes (species, disease, ...)
  // appear here with no code change. Species is just another classification
  // measurement in `schema.measurements` — no special-casing needed for it.
  const [schema, setSchema] = useState<SchemaDoc>(DEFAULT_SCHEMA);
  // `files` are what actually get uploaded (compressed once ready); `previews`
  // are shown immediately from the raw picked files so there's no wait before
  // the admin sees what they selected. The labels below (classValues /
  // numberValues / notes) are shared across every file in `files` — bulk
  // upload applies one set of labels to the whole batch.
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<{ url: string; name: string }[]>([]);
  const [preparing, setPreparing] = useState(false);

  // One value per measurement, keyed by measurement key. Classification ->
  // selected class name; regression -> raw number-input string (blank =
  // unset, falls back to the ML pipeline's anchor); segmentation -> an
  // uploaded mask file.
  const [classValues, setClassValues] = useState<Record<string, string>>({});
  const [numberValues, setNumberValues] = useState<Record<string, string>>({});
  const [maskFiles, setMaskFiles] = useState<Record<string, File | null>>({});

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
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const response = await fetch("/api/member/schema");
      const payload = await response.json().catch(() => null);
      if (!cancelled && response.ok && payload?.schema) applyDefaults(payload.schema as SchemaDoc);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Tracks the live preview URLs so they can be revoked on unmount without
  // re-running (and mis-revoking still-shown URLs) on every add/remove.
  const previewsRef = useRef(previews);
  useEffect(() => {
    previewsRef.current = previews;
  }, [previews]);
  useEffect(() => {
    return () => {
      for (const p of previewsRef.current) URL.revokeObjectURL(p.url);
    };
  }, []);

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(event.target.files ?? []);
    event.target.value = ""; // allow re-selecting the same file(s)
    if (picked.length === 0) return;

    // Replaces any previous selection (this isn't an "append" — picking again
    // starts a fresh batch), so revoke the outgoing preview URLs up front.
    for (const p of previews) URL.revokeObjectURL(p.url);
    setPreviews(picked.map((f) => ({ url: URL.createObjectURL(f), name: f.name })));
    setFiles(picked);
    setPreparing(true);
    try {
      setFiles(await Promise.all(picked.map(prepareForUpload)));
    } finally {
      setPreparing(false);
    }
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => {
      const removed = prev[index];
      if (removed) URL.revokeObjectURL(removed.url);
      return prev.filter((_, i) => i !== index);
    });
  }

  function resetForm() {
    for (const p of previews) URL.revokeObjectURL(p.url);
    setFiles([]);
    setPreviews([]);
    setNotes("");
    setNumberValues({});
    setMaskFiles({});
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (files.length === 0) {
      setError("Choose at least one photo first.");
      return;
    }
    if (preparing) {
      setError("Still preparing the photo(s) — try again in a moment.");
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
    for (const f of files) formData.append("files", f);
    formData.append("measurements", JSON.stringify(measurements));
    // A segmentation mask is specific to a single image's pixels, so it only
    // makes sense — and is only offered in the UI below — when exactly one
    // photo is selected.
    if (files.length === 1) {
      for (const m of schema.measurements) {
        if (m.type !== "segmentation" || !measurementApplies(m, classValues)) continue;
        const maskFile = maskFiles[m.key];
        if (maskFile) formData.append(`mask:${m.key}`, maskFile);
      }
    }
    formData.append("notes", notes);

    try {
      const response = await fetch("/api/member/dataset", { method: "POST", body: formData });
      const payload = await response.json().catch(() => null);
      // A per-file failure (e.g. one bad upload in a batch) still comes back
      // with `results` and a non-2xx status; only throw for errors that never
      // got to per-file processing (auth, invalid 'measurements', ...).
      if (!response.ok && !payload?.results) {
        throw new Error(payload?.error ?? `Upload failed (HTTP ${response.status}).`);
      }
      const results = (payload?.results ?? []) as { file: string; id?: string; error?: string }[];
      const failed = results.filter((r) => r.error);
      if (failed.length > 0) {
        setError(
          failed.length === results.length
            ? `All uploads failed: ${failed[0]?.error}`
            : `${failed.length} of ${results.length} photo(s) failed: ${failed.map((f) => `${f.file} (${f.error})`).join("; ")}`
        );
      }
      if (failed.length < results.length) {
        resetForm();
        onUploaded();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AdminCard className="p-5">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <h2 className={sectionHeadingClass}>Label new photos</h2>

        <AdminField label="Photos">
          <div className="flex flex-col gap-3">
            {previews.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {previews.map((p, i) => (
                  <div key={p.url} className="group relative h-20 w-20 flex-shrink-0">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={p.url} alt={p.name} className="h-20 w-20 border border-zinc-200 object-cover" />
                    <button
                      type="button"
                      onClick={() => removeFile(i)}
                      aria-label={`Remove ${p.name}`}
                      className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full border border-zinc-300 bg-white text-xs text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-100 group-hover:opacity-100"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex items-center gap-3">
              <label className="inline-flex w-fit cursor-pointer items-center rounded-sm border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100">
                {previews.length > 0 ? "Choose different photos" : "Choose photos"}
                <input type="file" accept="image/*" multiple onChange={handleFileChange} className="hidden" />
              </label>
              {previews.length > 0 && (
                <span className="text-xs text-zinc-500">
                  {previews.length} photo{previews.length === 1 ? "" : "s"} selected
                </span>
              )}
              {preparing && <span className="text-xs text-zinc-500">Preparing image{previews.length === 1 ? "" : "s"}…</span>}
            </div>
            {previews.length > 1 && (
              <p className="text-xs text-zinc-500">The labels below will be applied to all {previews.length} photos.</p>
            )}
          </div>
        </AdminField>

        {/* One control per schema measurement, in schema order (species,
            health status, disease, colour, the lab metrics, ...). A
            measurement with an applies_when that isn't satisfied by the
            current selections (e.g. disease_severity when disease ==
            NoDisease, or anything gated on seaweed_presence == Yes) is
            hidden. */}
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
                      {c.name}
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

          // segmentation — a mask is specific to one image's pixels, so it's
          // only offered when uploading a single photo.
          if (files.length > 1) return null;
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

        <AdminField label="Notes">
          <AdminTextarea value={notes} onChange={(event) => setNotes(event.target.value)} rows={2} />
        </AdminField>

        {error && <p className="text-sm text-rose-600">{error}</p>}

        <AdminButton type="submit" disabled={submitting || preparing} className="self-start">
          {submitting
            ? "Uploading…"
            : files.length > 1
              ? `Add ${files.length} photos to dataset`
              : "Add to dataset"}
        </AdminButton>
      </form>
    </AdminCard>
  );
}
