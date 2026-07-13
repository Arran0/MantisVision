import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { getFileRaw } from "@/lib/github";

const CONTENT_TYPES: Record<string, string> = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
  gif: "image/gif",
};

// Staged photos live on a GitHub branch, not a public bucket, so the admin
// table can't just point <img> at a raw GitHub URL (would need the token in
// the browser). This route fetches the blob server-side with the token and
// streams it back to the already-authenticated admin session instead.
export async function GET(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const path = request.nextUrl.searchParams.get("path");
  if (!path || path.includes("..")) {
    return NextResponse.json({ error: "Missing or invalid 'path'." }, { status: 400 });
  }

  const buffer = await getFileRaw(`ml/dataset_incoming/${path}`);
  if (!buffer) {
    return NextResponse.json({ error: "Not found." }, { status: 404 });
  }

  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const contentType = CONTENT_TYPES[ext] ?? "application/octet-stream";

  return new NextResponse(new Uint8Array(buffer), {
    headers: { "Content-Type": contentType, "Cache-Control": "private, max-age=300" },
  });
}
