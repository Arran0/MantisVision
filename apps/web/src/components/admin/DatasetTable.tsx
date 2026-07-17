"use client";

import { useEffect, useState } from "react";
import type { TrainingImage } from "@/lib/types";
import { AdminButton, AdminCard } from "@/components/admin/ui";

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

export function DatasetTable({
  images,
  loading,
  loadingMore = false,
  total = 0,
  hasMore = false,
  onLoadMore,
}: {
  images: TrainingImage[];
  loading: boolean;
  loadingMore?: boolean;
  total?: number;
  hasMore?: boolean;
  onLoadMore?: () => void;
}) {
  const [preview, setPreview] = useState<TrainingImage | null>(null);

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
        </div>
        <div>
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
    </div>
  );
}
