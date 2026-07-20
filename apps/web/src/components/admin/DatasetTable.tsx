"use client";

import { useEffect, useState } from "react";
import type { TrainingImage } from "@/lib/types";
import { measurementApplies, type SchemaDoc } from "@/lib/schema";
import {
  AdminButton,
  AdminCard,
  AdminField,
  AdminInput,
  AdminSelect,
  AdminTextarea,
} from "@/components/admin/ui";

type ImageEdit = { id: string; measurements: Record<string, string | number>; notes: string | null; species: string | null; colour: string | null };

function measurementSummary(measurements: Record<string, string | number>): string {
  const entries = Object.entries(measurements);
  if (entries.length === 0) return "—";
  return entries.map(([key, value]) => `${key}: ${value}`).join(" · ");
}

// Full-image preview shown when a row's thumbnail is clicked. Closes on
// backdrop click or Escape. Square panel — no rounded corners.
function ImagePreview({ image, onClose }: { image: TrainingImage; onClose: () => void }) {
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-dewberry-900/70 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden border border-zinc-200 bg-white"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-2.5">
          <span className="truncate text-sm font-medium italic text-zinc-800">{image.species ?? "Unlabeled"}</span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-sm px-2 py-1 text-sm text-zinc-500 hover:bg-zinc-100"
            aria-label="Close preview"
          >
            ✕
          </button>
        </div>
        <div className="flex items-center justify-center overflow-auto bg-zinc-50 p-3">
          {image.thumbnailUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={image.thumbnailUrl}
              alt={image.species ?? "Training image"}
              className="max-h-[60vh] w-auto object-contain"
            />
          ) : (
            <div className="flex h-48 w-full items-center justify-center text-sm text-zinc-400">
              Preview unavailable
            </div>
          )}
        </div>
        <div className="border-t border-zinc-200 px-4 py-3 text-xs text-zinc-600">
          <p>{measurementSummary(image.measurements)}</p>
          {image.notes && <p className="mt-1 text-zinc-500">{image.notes}</p>}
        </div>
      </div>
    </div>
  );
}

// Quick-edit form for an already-uploaded photo's feature-column
// (measurement) values and notes — fixing a mislabel doesn't require
// re-uploading the image. Segmentation masks aren't editable here (they'd
// need a new file upload); every classification/regression measurement is.
function EditImageModal({
  image,
  schema,
  onClose,
  onSaved,
}: {
  image: TrainingImage;
  schema: SchemaDoc;
  onClose: () => void;
  onSaved: (updated: ImageEdit) => void;
}) {
  const [classValues, setClassValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const m of schema.measurements) {
      if (m.type !== "classification") continue;
      const value = image.measurements[m.key];
      init[m.key] = typeof value === "string" ? value : m.classes?.[0]?.name ?? "";
    }
    return init;
  });
  const [numberValues, setNumberValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const m of schema.measurements) {
      if (m.type !== "regression") continue;
      const value = image.measurements[m.key];
      init[m.key] = typeof value === "number" ? String(value) : "";
    }
    return init;
  });
  const [notes, setNotes] = useState(image.notes ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function handleSave(event: React.FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);

    // Start from the existing values (keeps segmentation mask paths, which
    // this form has no controls for) and overwrite classification/regression
    // keys with the edited values.
    const measurements: Record<string, string | number> = { ...image.measurements };
    for (const m of schema.measurements) {
      if (!measurementApplies(m, classValues)) {
        delete measurements[m.key];
        continue;
      }
      if (m.type === "classification") {
        const value = classValues[m.key];
        if (value) measurements[m.key] = value;
        else delete measurements[m.key];
      } else if (m.type === "regression") {
        const raw = numberValues[m.key];
        if (raw !== undefined && raw.trim() !== "") {
          const num = Number(raw);
          if (Number.isFinite(num)) measurements[m.key] = num;
        } else {
          delete measurements[m.key];
        }
      }
    }

    try {
      const response = await fetch("/api/member/dataset", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: image.id, measurements, notes }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? `Update failed (HTTP ${response.status}).`);
      onSaved(payload as ImageEdit);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-dewberry-900/70 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden border border-zinc-200 bg-white"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-2.5">
          <span className="text-sm font-medium text-zinc-800">Edit labels</span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-sm px-2 py-1 text-sm text-zinc-500 hover:bg-zinc-100"
            aria-label="Close editor"
          >
            ✕
          </button>
        </div>
        <form onSubmit={handleSave} className="flex flex-col gap-4 overflow-y-auto p-4">
          {schema.measurements.map((m) => {
            if (!measurementApplies(m, classValues)) return null;

            if (m.type === "classification") {
              return (
                <AdminField key={m.key} label={m.label}>
                  <AdminSelect
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

            return null; // segmentation masks aren't editable from this quick-edit form
          })}

          <AdminField label="Notes">
            <AdminTextarea value={notes} onChange={(event) => setNotes(event.target.value)} rows={2} />
          </AdminField>

          {error && <p className="text-sm text-rose-600">{error}</p>}

          <div className="flex justify-end gap-2">
            <AdminButton type="button" variant="secondary" onClick={onClose}>
              Cancel
            </AdminButton>
            <AdminButton type="submit" disabled={saving}>
              {saving ? "Saving…" : "Save"}
            </AdminButton>
          </div>
        </form>
      </div>
    </div>
  );
}

export function DatasetTable({
  images,
  loading,
  loadingMore = false,
  total = 0,
  hasMore = false,
  onLoadMore,
  onImageUpdated,
}: {
  images: TrainingImage[];
  loading: boolean;
  loadingMore?: boolean;
  total?: number;
  hasMore?: boolean;
  onLoadMore?: () => void;
  onImageUpdated?: (updated: ImageEdit) => void;
}) {
  const [preview, setPreview] = useState<TrainingImage | null>(null);
  const [editing, setEditing] = useState<TrainingImage | null>(null);
  const [schema, setSchema] = useState<SchemaDoc | null>(null);

  // Fetched lazily once (not per-row) so opening the first edit form doesn't
  // wait, and later edits reuse the same schema.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const response = await fetch("/api/member/schema");
      const payload = await response.json().catch(() => null);
      if (!cancelled && response.ok && payload?.schema) setSchema(payload.schema as SchemaDoc);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <p className="text-sm text-zinc-500">Loading…</p>;
  }
  if (images.length === 0) {
    return <p className="text-sm text-zinc-500">No labeled photos yet.</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      <AdminCard className="overflow-hidden">
        <div className="flex bg-dewberry-900 px-4 py-2.5">
          <span className="w-16 flex-shrink-0 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Photo</span>
          <span className="flex-1 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Species</span>
          <span className="hidden flex-[2] text-[10px] font-bold uppercase tracking-widest text-zinc-400 sm:block">
            Measurements
          </span>
          <span className="w-24 flex-shrink-0 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Status</span>
          <span className="hidden w-28 flex-shrink-0 text-right text-[10px] font-bold uppercase tracking-widest text-zinc-400 sm:block">
            Added
          </span>
          <span className="w-14 flex-shrink-0 text-right text-[10px] font-bold uppercase tracking-widest text-zinc-400">
            Edit
          </span>
        </div>
        <div className="max-h-[30rem] overflow-y-auto">
          {images.map((image) => (
            <div
              key={image.id}
              className="flex items-center gap-0 border-b border-zinc-100 px-4 py-2.5 last:border-0 hover:bg-zinc-50"
            >
              <div className="w-16 flex-shrink-0">
                {image.thumbnailUrl ? (
                  <button
                    type="button"
                    onClick={() => setPreview(image)}
                    className="block h-12 w-12 overflow-hidden border border-zinc-200 transition-opacity hover:opacity-80"
                    aria-label="Preview photo"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={image.thumbnailUrl} alt="" className="h-full w-full object-cover" />
                  </button>
                ) : (
                  <div className="h-12 w-12 border border-zinc-200 bg-zinc-100" />
                )}
              </div>
              <span className="flex-1 truncate text-sm italic text-zinc-800">{image.species ?? "—"}</span>
              <span className="hidden flex-[2] truncate text-xs text-zinc-600 sm:block">
                {measurementSummary(image.measurements)}
              </span>
              <span className="w-24 flex-shrink-0 text-xs text-zinc-600">{image.status}</span>
              <span className="hidden w-28 flex-shrink-0 text-right text-xs text-zinc-400 sm:block">
                {new Date(image.createdAt).toLocaleDateString()}
              </span>
              <span className="w-14 flex-shrink-0 text-right">
                <button
                  type="button"
                  onClick={() => setEditing(image)}
                  disabled={!schema}
                  className="text-xs font-medium text-dewberry-700 underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:text-zinc-300"
                >
                  Edit
                </button>
              </span>
            </div>
          ))}
        </div>
      </AdminCard>

      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-zinc-500">
          Showing {images.length}
          {total > 0 ? ` of ${total}` : ""} photo{images.length === 1 ? "" : "s"}
        </p>
        {hasMore && onLoadMore && (
          <AdminButton type="button" variant="secondary" onClick={onLoadMore} disabled={loadingMore}>
            {loadingMore ? "Loading…" : "Load more"}
          </AdminButton>
        )}
      </div>

      {preview && <ImagePreview image={preview} onClose={() => setPreview(null)} />}
      {editing && schema && (
        <EditImageModal
          image={editing}
          schema={schema}
          onClose={() => setEditing(null)}
          onSaved={(updated) => onImageUpdated?.(updated)}
        />
      )}
    </div>
  );
}
