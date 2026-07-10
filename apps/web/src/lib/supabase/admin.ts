import { createClient as createSupabaseClient } from "@supabase/supabase-js";

// Service-role Supabase client: bypasses RLS entirely. Import this only from
// /api/admin/* route handlers, after requireAdmin() has already verified the
// caller — never from a Server/Client Component or anything that ships to
// the browser.
export function createAdminClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceRoleKey) {
    throw new Error("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.");
  }
  return createSupabaseClient(url, serviceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false },
  });
}
