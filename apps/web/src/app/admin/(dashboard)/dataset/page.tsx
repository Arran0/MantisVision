"use client";

import { useCallback, useEffect, useState } from "react";
import type { TrainingImage } from "@/lib/types";
import { DatasetUploadForm } from "@/components/admin/DatasetUploadForm";
import { DatasetTable } from "@/components/admin/DatasetTable";

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
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Dataset</h1>
        <p className="mt-1 text-sm text-slate-600">
          Photos labeled here feed the training dataset used by a retrain run.
        </p>
      </div>
      <DatasetUploadForm onUploaded={refresh} />
      <DatasetTable images={images} loading={loading} />
    </div>
  );
}
