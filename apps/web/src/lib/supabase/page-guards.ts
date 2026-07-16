import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import type { Role } from "@/lib/roles";

export type DashboardUser = { id: string; email: string | null; role: Role };

// Resolves the signed-in user's account level for Server Components. Returns
// null when there is no session at all (the layout turns that into a login
// redirect).
export async function getDashboardUser(): Promise<DashboardUser | null> {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", user.id)
    .single();

  return { id: user.id, email: user.email ?? null, role: (profile?.role ?? "viewer") as Role };
}

// Guard for admin-only pages (Structure, Retrain, Team). A contributor who
// navigates straight to one of these URLs is bounced back to their home tab
// rather than shown a page they can't use.
export async function requireAdminPage(): Promise<DashboardUser> {
  const user = await getDashboardUser();
  if (!user) redirect("/admin/login");
  if (user.role !== "admin") redirect("/admin/home");
  return user;
}
