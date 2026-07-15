import type { SupabaseClient } from "@supabase/supabase-js";
import { DEFAULT_SCHEMA, normalizeSchemaDoc, type SchemaDoc } from "@/lib/schema";

// Import this only from /api/admin/* route handlers (it uses the service-role
// client). Never import it from a Client Component.

// Fetches the active (most recent) measurement_schema document via a
// service-role client. Falls back to DEFAULT_SCHEMA when no row exists yet
// (e.g. before the seed migration has run), so callers always get a usable
// schema. Normalized so a row saved before applies_when became a list (a
// single condition object) still loads correctly.
export async function getActiveSchema(admin: SupabaseClient): Promise<SchemaDoc> {
  const { data, error } = await admin
    .from("measurement_schema")
    .select("doc")
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error || !data?.doc) return DEFAULT_SCHEMA;
  return normalizeSchemaDoc(data.doc as SchemaDoc);
}
