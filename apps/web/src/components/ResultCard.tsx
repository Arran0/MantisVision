import type { PredictionResult } from "@/lib/types";

const HEALTH_COLORS: Record<string, string> = {
  Healthy: "bg-green-100 text-green-800",
  Moderate: "bg-yellow-100 text-yellow-800",
  Low: "bg-orange-100 text-orange-800",
  Decay: "bg-amber-100 text-amber-900",
  Dead: "bg-slate-200 text-slate-800",
  Predator: "bg-red-100 text-red-800",
  Disease: "bg-purple-100 text-purple-800",
};

export function ResultCard({ result }: { result: PredictionResult }) {
  const badgeClass = HEALTH_COLORS[result.health] ?? "bg-slate-100 text-slate-800";

  return (
    <div className="flex flex-col gap-4 rounded-2xl bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500">Species</p>
          <p className="text-lg font-medium italic">{result.species}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${badgeClass}`}>
          {result.health}
        </span>
      </div>

      <div>
        <p className="text-sm text-slate-500">Confidence</p>
        <p className="text-lg font-medium">{(result.confidence * 100).toFixed(1)}%</p>
      </div>

      <div>
        <p className="text-sm text-slate-500">Explanation</p>
        <p>{result.explanation}</p>
      </div>

      <div>
        <p className="text-sm text-slate-500">Recommendation</p>
        <p>{result.recommendation}</p>
      </div>

      {result.gradcamPngBase64 && (
        <div>
          <p className="mb-2 text-sm text-slate-500">Why the model said this (Grad-CAM)</p>
          <img
            src={`data:image/png;base64,${result.gradcamPngBase64}`}
            alt="Grad-CAM heatmap"
            className="w-full rounded-xl"
          />
        </div>
      )}
    </div>
  );
}
