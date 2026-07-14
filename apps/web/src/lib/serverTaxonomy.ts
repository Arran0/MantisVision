import type { SupabaseClient } from "@supabase/supabase-js";
import { DEFAULT_TAXONOMY, type TaxonomyDoc } from "@/lib/taxonomy";

// Import this only from /api/admin/* route handlers (it uses the service-role
// client). Never import it from a Client Component.

// Fetches the active (most recent) dataset_taxonomy document via a service-role
// client. Falls back to DEFAULT_TAXONOMY when no row exists yet (e.g. before
// the seed migration has run), so callers always get a usable taxonomy.
export async function getActiveTaxonomy(admin: SupabaseClient): Promise<TaxonomyDoc> {
  const { data, error } = await admin
    .from("dataset_taxonomy")
    .select("doc")
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error || !data?.doc) return DEFAULT_TAXONOMY;
  return data.doc as TaxonomyDoc;
}
