import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { getActiveSchema, invalidateActiveSchemaCache } from "@/lib/serverSchema";
import { validateSchema, type SchemaDoc } from "@/lib/schema";

// GET  -> the active (most recent) measurement schema document.
// PUT  -> validate a candidate document and, if valid, append it as a new
//         active version (append-only, mirroring model_runs' audit trail).
//
// Editing the schema takes effect for new labeling and validation
// immediately; it reaches the *model* on the next retrain (the workflow
// exports the active schema into the run) and the *serving copy* when that
// run's checkpoint is promoted (the checkpoint carries the schema it was
// trained with).

export async function GET() {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const admin = createAdminClient();
  const schema = await getActiveSchema(admin);
  return NextResponse.json({ schema });
}

export async function PUT(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const doc = body?.schema as SchemaDoc | undefined;

  const problem = validateSchema(doc);
  if (problem) {
    return NextResponse.json({ error: problem }, { status: 400 });
  }

  const admin = createAdminClient();
  const { error } = await admin
    .from("measurement_schema")
    .insert({ created_by: auth.context.userId, doc });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  invalidateActiveSchemaCache();
  return NextResponse.json({ ok: true, schema: doc });
}
