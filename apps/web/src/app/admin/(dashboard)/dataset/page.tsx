"use client";

import { useCallback, useEffect, useState } from "react";
import type { TrainingImage } from "@/lib/types";
import { DatasetUploadForm } from "@/components/admin/DatasetUploadForm";
import { DatasetTable } from "@/components/admin/DatasetTable";
import { AdminPageHeader } from "@/components/admin/ui";

export default function DatasetPage() {
  const [images, setImages] = useState<TrainingImage[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/admin/dataset");
      const payload = await response.json().catch(() => null);
      setImages(payload?.images ?? []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader title="Dataset" subtitle="Photos labeled here feed the training dataset used by a retrain run." />
      <DatasetUploadForm onUploaded={refresh} />
      <DatasetTable images={images} loading={loading} />
    </div>
  );
}
