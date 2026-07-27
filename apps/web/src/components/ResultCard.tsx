import type { MeasurementResult, PredictionResult } from "@/lib/types";

// Fully schema-driven: every measurement the API returned is rendered by its
// own `type`, in the order the backend sent them (schema order) — nothing
// here is keyed off a specific measurement name like "species" or
// "health_status". A measurement gated off by applies_when arrives with a
// null value and is simply skipped.

function ConfidenceBadge({ confidence }: { confidence: number | null }) {
  if (confidence === null) return null;
  return <span className="text-xs font-medium text-slate-400">{Math.round(confidence * 100)}% confidence</span>;
}

function ClassificationRow({ measurement }: { measurement: MeasurementResult }) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{measurement.label}</p>
        <ConfidenceBadge confidence={measurement.confidence} />
      </div>
      <p className="mt-1 text-lg font-semibold text-slate-900">{measurement.value}</p>
      {measurement.explanation && <p className="mt-1 text-sm leading-relaxed text-slate-600">{measurement.explanation}</p>}
      {measurement.recommendation && <p className="mt-1 text-sm leading-relaxed text-slate-500">{measurement.recommendation}</p>}
      {measurement.gradcamPngBase64 && (
        <div className="mt-3">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            Why the model said this (Grad-CAM)
          </p>
          <img
            src={`data:image/png;base64,${measurement.gradcamPngBase64}`}
            alt={`${measurement.label} Grad-CAM heatmap`}
            className="w-full rounded-2xl"
          />
        </div>
      )}
    </div>
  );
}

function RegressionRow({ measurement }: { measurement: MeasurementResult }) {
  if (typeof measurement.value !== "number") return null;
  const min = measurement.min ?? 0;
  const max = measurement.max ?? 100;
  const pct = max > min ? Math.max(0, Math.min(100, Math.round(((measurement.value - min) / (max - min)) * 100))) : 0;

  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{measurement.label}</p>
        <p className="text-sm font-semibold text-slate-700">
          {measurement.value}
          {measurement.unit ? ` ${measurement.unit}` : ""}
        </p>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${pct}%`, backgroundImage: "linear-gradient(90deg, #ff7a1a, #1a7ae0)" }}
        />
      </div>
      {measurement.explanation && <p className="mt-1 text-sm leading-relaxed text-slate-600">{measurement.explanation}</p>}
      {measurement.recommendation && <p className="mt-1 text-sm leading-relaxed text-slate-500">{measurement.recommendation}</p>}
    </div>
  );
}

function SegmentationRow({ measurement }: { measurement: MeasurementResult }) {
  const coverage = measurement.coverage;
  if (!coverage) return null;
  const colors = measurement.segColors ?? {};

  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{measurement.label}</p>
      <div className="mt-2 space-y-1.5">
        {Object.entries(coverage).map(([className, pct]) => (
          <div key={className} className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: colors[className] ?? "#888888" }}
            />
            <span className="w-24 shrink-0 truncate text-xs text-slate-600">{className}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full"
                style={{ width: `${pct}%`, backgroundColor: colors[className] ?? "#888888" }}
              />
            </div>
            <span className="w-10 shrink-0 text-right text-xs text-slate-500">{pct}%</span>
          </div>
        ))}
      </div>
      {measurement.maskPngBase64 && (
        <img
          src={`data:image/png;base64,${measurement.maskPngBase64}`}
          alt={`${measurement.label} mask overlay`}
          className="mt-3 w-full rounded-2xl"
        />
      )}
    </div>
  );
}

// A measurement gated off by applies_when has value === null; segmentation
// results carry their data in `coverage` instead of `value`.
function hasResult(measurement: MeasurementResult): boolean {
  return measurement.type === "segmentation" ? measurement.coverage !== null : measurement.value !== null;
}

export function ResultCard({ result }: { result: PredictionResult }) {
  const entries = Object.entries(result.measurements).filter(([, m]) => hasResult(m));

  if (entries.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <span className="text-lg font-semibold text-slate-700">No result</span>
        <p className="max-w-xs text-sm text-slate-600">
          The model didn&rsquo;t return any applicable measurement for this image.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {entries.map(([key, measurement]) => (
        <div key={key}>
          {measurement.type === "classification" ? (
            <ClassificationRow measurement={measurement} />
          ) : measurement.type === "regression" ? (
            <RegressionRow measurement={measurement} />
          ) : (
            <SegmentationRow measurement={measurement} />
          )}
        </div>
      ))}
    </div>
  );
}
