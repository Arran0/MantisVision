import { NextRequest, NextResponse } from "next/server";
import { ML_API_URL } from "@/lib/config";

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

  let upstreamResponse: Response;
  try {
    upstreamResponse = await fetch(`${ML_API_URL}/predict`, {
      method: "POST",
      body: upstreamForm,
    });
  } catch {
    return NextResponse.json(
      { error: `Could not reach the inference API at ${ML_API_URL}. Is it running?` },
      { status: 502 }
    );
  }

  const payload = await upstreamResponse.json();
  if (!upstreamResponse.ok) {
    return NextResponse.json(payload, { status: upstreamResponse.status });
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
