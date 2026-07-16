import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import type { Role } from "@/lib/roles";

export type AuthContext = { userId: string; email: string | null; role: Role };
// Kept as an alias so existing imports/annotations keep compiling.
export type AdminContext = AuthContext;
export type RequireRoleResult =
  | { ok: true; context: AuthContext }
  | { ok: false; response: NextResponse };
export type RequireAdminResult = RequireRoleResult;

// middleware.ts only checks "is there a session" for /api/admin/*. This is the
// actual authorization check — "does this session hold one of these roles" —
// and must be called at the top of every admin route handler, since
// middleware is a coarse first line of defense, not the source of truth.
export async function requireRole(roles: Role[]): Promise<RequireRoleResult> {
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

  const role = (profile?.role ?? null) as Role | null;
  if (!role || !roles.includes(role)) {
    return { ok: false, response: NextResponse.json({ error: "Forbidden" }, { status: 403 }) };
  }

  return { ok: true, context: { userId: user.id, email: user.email ?? null, role } };
}

// Admin-only routes (schema, retrain, team management).
export function requireAdmin(): Promise<RequireRoleResult> {
  return requireRole(["admin"]);
}

// Routes a contributor may also use (dataset labeling).
export function requireContributor(): Promise<RequireRoleResult> {
  return requireRole(["admin", "contributor"]);
}
