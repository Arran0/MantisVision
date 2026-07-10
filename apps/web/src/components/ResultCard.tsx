import type { PredictionResult } from "@/lib/types";

const HEALTH_COLORS: Record<string, string> = {
  Healthy: "bg-seaweed-100 text-seaweed-700",
  Moderate: "bg-yellow-100 text-yellow-800",
  Low: "bg-coral-500/15 text-coral-600",
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 leading-relaxed text-slate-800">{children}</p>
    </div>
  );
}

function Meter({ label, value, colorFrom, colorTo }: { label: string; value: number; colorFrom: string; colorTo: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
        <p className="text-sm font-semibold text-slate-700">{pct}%</p>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${pct}%`, backgroundImage: `linear-gradient(90deg, ${colorFrom}, ${colorTo})` }}
        />
      </div>
    </div>
  );
}

export function ResultCard({ result }: { result: PredictionResult }) {
  const confidence = Math.round(result.confidence * 100);

  // No-seaweed state: the Background class fired, so we deliberately don't
  // show any health assessment — just say nothing was detected.
  if (!result.isSeaweed) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <span className="text-lg font-semibold text-slate-700">No seaweed detected</span>
        <p className="max-w-xs text-sm text-slate-600">{result.explanation}</p>
        <p className="max-w-xs text-sm text-slate-500">{result.recommendation}</p>
        <span className="text-xs text-slate-400">Confidence {confidence}%</span>
      </div>
    );
  }

  const level = result.health ?? "—";
  const badgeClass = HEALTH_COLORS[level] ?? "bg-slate-100 text-slate-700";

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Species</p>
          <p className="mt-1 text-2xl font-semibold italic text-slate-900">{result.species}</p>
          <p className="mt-1 text-sm text-slate-500">
            Condition: <span className="font-medium text-slate-700">{result.condition}</span>
            {result.diseaseSubtype ? ` · ${result.diseaseSubtype}` : ""}
          </p>
        </div>
        <span className={`shrink-0 rounded-full px-3 py-1 text-sm font-semibold ${badgeClass}`}>{level}</span>
      </div>

      {result.healthScore !== null && (
        <Meter label="Health score" value={result.healthScore} colorFrom="#ff7a1a" colorTo="#16a34a" />
      )}

      <Meter label="Confidence" value={confidence} colorFrom="#ff7a1a" colorTo="#1a7ae0" />

      {(result.driedPct !== null || result.decayedPct !== null) && (
        <div className="grid gap-4 sm:grid-cols-2">
          {result.driedPct !== null && (
            <Meter label="Dried extent" value={result.driedPct} colorFrom="#cbd5e1" colorTo="#64748b" />
          )}
          {result.decayedPct !== null && (
            <Meter label="Decayed extent" value={result.decayedPct} colorFrom="#fcd34d" colorTo="#b45309" />
          )}
        </div>
      )}

      <div className="grid gap-6 sm:grid-cols-2">
        <Field label="Explanation">{result.explanation}</Field>
        <Field label="Recommendation">{result.recommendation}</Field>
      </div>

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
