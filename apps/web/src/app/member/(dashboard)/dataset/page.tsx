"use client";

import { useCallback, useEffect, useState } from "react";
import type { TrainingImage } from "@/lib/types";
import type { SchemaDoc } from "@/lib/schema";
import { DatasetUploadForm } from "@/components/admin/DatasetUploadForm";
import { DatasetTable } from "@/components/admin/DatasetTable";
import { DatasetFilters, EMPTY_FILTERS, buildFiltersParam, type DatasetFilterState } from "@/components/admin/DatasetFilters";
import { AdminPageHeader } from "@/components/admin/ui";

type ImageEdit = {
  id: string;
  measurements: Record<string, string | number>;
  notes: string | null;
  species: string | null;
  colour: string | null;
  split: "train" | "validation" | "test" | null;
};

export default function DatasetPage() {
  const [images, setImages] = useState<TrainingImage[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  // `loading` covers the first page (fresh list); `loadingMore` covers
  // appending a subsequent page so the two states don't fight over the UI.
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [schema, setSchema] = useState<SchemaDoc | null>(null);
  const [filters, setFilters] = useState<DatasetFilterState>(EMPTY_FILTERS);

  // Fetched once for the filter panel's schema-driven controls (same schema
  // DatasetTable's edit form loads independently).
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

  // Loads one page of PAGE_SIZE, applying the current filter state. page 1
  // replaces the list (used on mount, after an upload, and whenever filters
  // change); later pages are appended. Keeps memory/DOM light on a big
  // dataset. Recreated whenever `filters` changes so the effect below
  // reloads page 1 with the new filter set.
  const loadPage = useCallback(
    async (nextPage: number) => {
      if (nextPage === 1) setLoading(true);
      else setLoadingMore(true);
      try {
        const query = new URLSearchParams({ page: String(nextPage) });
        const filtersParam = buildFiltersParam(filters);
        if (filtersParam) query.set("filters", filtersParam);
        const response = await fetch(`/api/member/dataset?${query.toString()}`);
        const payload = await response.json().catch(() => null);
        const batch: TrainingImage[] = payload?.images ?? [];
        setImages((prev) => (nextPage === 1 ? batch : [...prev, ...batch]));
        setPage(nextPage);
        setTotal(payload?.total ?? batch.length);
        setHasMore(Boolean(payload?.hasMore));
      } finally {
        if (nextPage === 1) setLoading(false);
        else setLoadingMore(false);
      }
    },
    [filters]
  );

  const refresh = useCallback(() => loadPage(1), [loadPage]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function handleImageUpdated(updated: ImageEdit) {
    setImages((prev) => prev.map((image) => (image.id === updated.id ? { ...image, ...updated } : image)));
  }

  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader title="Dataset" subtitle="Photos labeled here feed the training dataset used by a retrain run." />
      <DatasetUploadForm onUploaded={refresh} />
      {schema && <DatasetFilters schema={schema} value={filters} onChange={setFilters} />}
      <DatasetTable
        images={images}
        loading={loading}
        loadingMore={loadingMore}
        total={total}
        hasMore={hasMore}
        onLoadMore={() => loadPage(page + 1)}
        onImageUpdated={handleImageUpdated}
      />
    </div>
  );
}
