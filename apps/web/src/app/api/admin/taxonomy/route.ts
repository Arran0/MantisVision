import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { getActiveTaxonomy } from "@/lib/serverTaxonomy";
import { validateTaxonomy, type TaxonomyDoc } from "@/lib/taxonomy";

// GET  -> the active (most recent) taxonomy document.
// PUT  -> validate a candidate document and, if valid, append it as a new
//         active version (append-only, mirroring model_runs' audit trail).
//
// Editing the taxonomy takes effect for new labeling and validation
// immediately; it reaches the *model* on the next retrain (the workflow
// exports the active taxonomy into the run) and the *serving copy* when that
// run's checkpoint is promoted.

export async function GET() {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const admin = createAdminClient();
  const taxonomy = await getActiveTaxonomy(admin);
  return NextResponse.json({ taxonomy });
}

export async function PUT(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const doc = body?.taxonomy as TaxonomyDoc | undefined;

  const problem = validateTaxonomy(doc);
  if (problem) {
    return NextResponse.json({ error: problem }, { status: 400 });
  }

  const admin = createAdminClient();
  const { error } = await admin
    .from("dataset_taxonomy")
    .insert({ created_by: auth.context.userId, doc });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  return NextResponse.json({ ok: true, taxonomy: doc });
}
