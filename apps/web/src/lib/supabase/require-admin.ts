import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

export type AdminContext = { userId: string; email: string | null };
export type RequireAdminResult =
  | { ok: true; context: AdminContext }
  | { ok: false; response: NextResponse };

// middleware.ts only checks "is there a session" for /api/admin/*. This is
// the actual authorization check — "is this session an admin" — and must be
// called at the top of every admin route handler, since middleware is a
// coarse first line of defense, not the source of truth.
export async function requireAdmin(): Promise<RequireAdminResult> {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { ok: false, response: NextResponse.json({ error: "Unauthorized" }, { status: 401 }) };
  }

  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", user.id)
    .single();

  if (profile?.role !== "admin") {
    return { ok: false, response: NextResponse.json({ error: "Forbidden" }, { status: 403 }) };
  }

  return { ok: true, context: { userId: user.id, email: user.email ?? null } };
}
