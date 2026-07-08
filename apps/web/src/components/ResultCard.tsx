import type { PredictionResult } from "@/lib/types";

const HEALTH_COLORS: Record<string, string> = {
  Healthy: "bg-seaweed-100 text-seaweed-700",
  Moderate: "bg-yellow-100 text-yellow-800",
  Low: "bg-coral-500/15 text-coral-600",
  Decay: "bg-amber-100 text-amber-900",
  Dried: "bg-slate-200 text-slate-700",
  Disease: "bg-purple-100 text-purple-800",
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-slate-800">{children}</p>
    </div>
  );
}

export function ResultCard({ result }: { result: PredictionResult }) {
  const badgeClass = HEALTH_COLORS[result.health] ?? "bg-slate-100 text-slate-700";
  const confidence = Math.round(result.confidence * 100);

  return (
    <div className="mv-card flex flex-col gap-5 p-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Species</p>
          <p className="mt-1 text-xl font-semibold italic text-slate-900">{result.species}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-sm font-semibold ${badgeClass}`}>
          {result.health}
        </span>
      </div>

      <div>
        <div className="flex items-center justify-between">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Confidence</p>
          <p className="text-sm font-semibold text-slate-700">{confidence}%</p>
        </div>
        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full"
            style={{
              width: `${confidence}%`,
              backgroundImage: "linear-gradient(90deg, #ff7a1a, #1a7ae0, #16a34a)",
            }}
          />
        </div>
      </div>

      <Field label="Explanation">{result.explanation}</Field>
      <Field label="Recommendation">{result.recommendation}</Field>

      {result.gradcamPngBase64 && (
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            Why the model said this (Grad-CAM)
          </p>
          <img
            src={`data:image/png;base64,${result.gradcamPngBase64}`}
            alt="Grad-CAM heatmap"
            className="w-full rounded-2xl"
          />
        </div>
      )}
    </div>
  );
}
