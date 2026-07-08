import { NextRequest, NextResponse } from "next/server";
import { ML_API_URL } from "@/lib/config";

// Abort the upstream call rather than letting a stuck inference service hold
// the request open until the platform kills the function.
const UPSTREAM_TIMEOUT_MS = 25_000;

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
    health: payload.health,
    confidence: payload.confidence,
    explanation: payload.explanation,
    recommendation: payload.recommendation,
    gradcamPngBase64: payload.gradcam_png_base64,
  });
}
