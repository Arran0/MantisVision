"use client";

import { useState } from "react";
import type { PredictionResult } from "@/lib/types";
import { ResultCard } from "@/components/ResultCard";

export function UploadCard() {
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error ?? payload.detail ?? "Prediction failed.");
      }

      setResult(payload as PredictionResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-6">
      <label className="flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-seaweed-500 bg-white p-8 text-center transition hover:bg-seaweed-50">
        <span className="text-lg font-medium text-seaweed-900">
          Photograph a Kappaphycus alvarezii specimen
        </span>
        <span className="text-sm text-slate-500">Tap to upload or take a photo</span>
        <input
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>

      {preview && (
        <img src={preview} alt="Uploaded specimen" className="max-h-80 w-full rounded-xl object-contain" />
      )}

      {loading && <p className="text-center text-slate-500">Analyzing...</p>}
      {error && <p className="text-center text-red-600">{error}</p>}
      {result && <ResultCard result={result} />}
    </div>
  );
}
