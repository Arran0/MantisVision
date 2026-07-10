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

  return NextResponse.json({
    species: payload.species,
    isSeaweed: payload.is_seaweed,
    condition: payload.condition,
    health: payload.health,
    healthScore: payload.health_score,
    confidence: payload.confidence,
    diseaseSubtype: payload.disease_subtype,
    driedPct: payload.dried_pct,
    decayedPct: payload.decayed_pct,
    explanation: payload.explanation,
    recommendation: payload.recommendation,
    gradcamPngBase64: payload.gradcam_png_base64,
  });
}
