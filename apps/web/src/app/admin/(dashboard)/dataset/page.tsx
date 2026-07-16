"use client";

import { useCallback, useEffect, useState } from "react";
import type { TrainingImage } from "@/lib/types";
import { DatasetUploadForm } from "@/components/admin/DatasetUploadForm";
import { DatasetTable } from "@/components/admin/DatasetTable";
import { AdminPageHeader } from "@/components/admin/ui";

export default function DatasetPage() {
  const [images, setImages] = useState<TrainingImage[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  // `loading` covers the first page (fresh list); `loadingMore` covers
  // appending a subsequent page so the two states don't fight over the UI.
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  // Loads one page of 15. page 1 replaces the list (used on mount and after an
  // upload); later pages are appended. Keeps memory/DOM light on a big dataset.
  const loadPage = useCallback(async (nextPage: number) => {
    if (nextPage === 1) setLoading(true);
    else setLoadingMore(true);
    try {
      const response = await fetch(`/api/admin/dataset?page=${nextPage}`);
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
  }, []);

  const refresh = useCallback(() => loadPage(1), [loadPage]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader title="Dataset" subtitle="Photos labeled here feed the training dataset used by a retrain run." />
      <DatasetUploadForm onUploaded={refresh} />
      <DatasetTable
        images={images}
        loading={loading}
        loadingMore={loadingMore}
        total={total}
        hasMore={hasMore}
        onLoadMore={() => loadPage(page + 1)}
      />
    </div>
  );
}
