import type { SupabaseClient } from "@supabase/supabase-js";
import { DEFAULT_SCHEMA, normalizeSchemaDoc, type SchemaDoc } from "@/lib/schema";

// Import this only from /api/member/* route handlers (it uses the service-role
// client). Never import it from a Client Component.

// The schema changes only when an admin explicitly edits it (via PUT
// /api/member/schema), yet every dataset label save was re-fetching it from
// the DB. Cache it in module scope (persists across requests on a warm
// serverless instance) and drop the cache as soon as it's edited, so
// "editing the schema takes effect immediately" still holds on the instance
// that served the edit. The short TTL is just a backstop for other warm
// instances that didn't see the invalidation.
const CACHE_TTL_MS = 30_000;
let cached: { schema: SchemaDoc; fetchedAt: number } | null = null;

export function invalidateActiveSchemaCache(): void {
  cached = null;
}

// Fetches the active (most recent) measurement_schema document via a
// service-role client. Falls back to DEFAULT_SCHEMA when no row exists yet
// (e.g. before the seed migration has run), so callers always get a usable
// schema. Normalized so a row saved before applies_when became a list (a
// single condition object) still loads correctly.
export async function getActiveSchema(admin: SupabaseClient): Promise<SchemaDoc> {
  if (cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) return cached.schema;

  const { data, error } = await admin
    .from("measurement_schema")
    .select("doc")
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error || !data?.doc) return DEFAULT_SCHEMA;

  const schema = normalizeSchemaDoc(data.doc as SchemaDoc);
  cached = { schema, fetchedAt: Date.now() };
  return schema;
}
