import { NextRequest, NextResponse } from "next/server";
import { ML_API_URL } from "@/lib/config";

// A free-tier inference host (e.g. Render) can cold-start for ~50s after
// idling. Allow the function to run long enough to wait one out instead of
// being killed at the platform's 10s default. 60s is the Vercel Hobby ceiling.
export const maxDuration = 60;

// Abort the upstream call rather than letting a stuck inference service hold
// the request open until the platform kills the function. Kept just under
// maxDuration so we still return a clean JSON error on a genuine hang.
const UPSTREAM_TIMEOUT_MS = 55_000;

// Thin proxy to the Python inference service so the browser never needs to
// know the ML service's address (and so CORS/auth can be layered on here
// later without touching the ML API itself).
export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const file = formData.get("file");

  if (!(file instanceof Blob)) {
    return NextResponse.json({ error: "Missing 'file' in form data." }, { status: 400 });
  }

  const upstreamForm = new FormData();
  upstreamForm.append("file", file, "upload.jpg");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  let upstreamResponse: Response;
  try {
    upstreamResponse = await fetch(`${ML_API_URL}/predict`, {
      method: "POST",
      body: upstreamForm,
      signal: controller.signal,
    });
  } catch (err) {
    const timedOut = err instanceof DOMException && err.name === "AbortError";
    return NextResponse.json(
      {
        error: timedOut
          ? `The inference API at ${ML_API_URL} didn't respond in time.`
          : `Could not reach the inference API at ${ML_API_URL}. Is it running and publicly reachable?`,
      },
      { status: 502 }
    );
  } finally {
    clearTimeout(timer);
  }

  const payload = await upstreamResponse.json().catch(() => null);
  if (!upstreamResponse.ok || !payload) {
    return NextResponse.json(
      payload ?? { error: `Inference API returned HTTP ${upstreamResponse.status}.` },
      { status: upstreamResponse.ok ? 502 : upstreamResponse.status }
    );
  }

  // Pass the schema-driven measurements map straight through, camelCasing
  // each entry's field names (not the measurement keys themselves — those
  // are admin-defined schema keys, not fixed field names).
  const rawMeasurements = (payload.measurements ?? {}) as Record<string, UpstreamMeasurement>;
  const measurements = Object.fromEntries(
    Object.entries(rawMeasurements).map(([key, m]) => [
      key,
      {
        type: m.type,
        label: m.label,
        value: m.value,
        confidence: m.confidence,
        explanation: m.explanation,
        recommendation: m.recommendation,
        unit: m.unit,
        min: m.min,
        max: m.max,
        coverage: m.coverage,
        segColors: m.seg_colors,
        maskPngBase64: m.mask_png_base64,
        gradcamPngBase64: m.gradcam_png_base64,
      },
    ])
  );

  return NextResponse.json({ measurements });
}

// Shape of a single entry in the ML API's `measurements` response map
// (snake_case, as FastAPI serializes it) — see ml/src/api/schemas.py's
// MeasurementResultResponse.
interface UpstreamMeasurement {
  type: string;
  label: string;
  value: string | number | null;
  confidence: number | null;
  explanation: string | null;
  recommendation: string | null;
  unit: string | null;
  min: number | null;
  max: number | null;
  coverage: Record<string, number> | null;
  seg_colors: Record<string, string> | null;
  mask_png_base64: string | null;
  gradcam_png_base64: string | null;
}
