import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";

// Promotion is intentionally NOT an automatic checkpoint swap. It records
// that an admin reviewed the run's metrics and approved it, then returns the
// exact steps to apply on the serving host (docs/DEPLOY_ML_API.md /
// docs/DEPLOY_HUGGINGFACE.md) — a deliberate manual last step, so a bad
// retrain can never silently reach production.
export async function POST(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const modelRunId = body?.modelRunId;
  if (typeof modelRunId !== "string") {
    return NextResponse.json({ error: "'modelRunId' is required." }, { status: 400 });
  }

  const admin = createAdminClient();
  const { data: run, error: fetchError } = await admin
    .from("model_runs")
    .select("id, status, checkpoint_url")
    .eq("id", modelRunId)
    .single();

  if (fetchError || !run) {
    return NextResponse.json({ error: "Run not found." }, { status: 404 });
  }
  if (run.status !== "completed" || !run.checkpoint_url) {
    return NextResponse.json(
      { error: "Only a completed run with a checkpoint can be promoted." },
      { status: 400 }
    );
  }

  const { error: updateError } = await admin
    .from("model_runs")
    .update({ promoted_at: new Date().toISOString(), promoted_by: auth.context.userId })
    .eq("id", modelRunId);

  if (updateError) {
    return NextResponse.json({ error: updateError.message }, { status: 502 });
  }

  return NextResponse.json({
    checkpointUrl: run.checkpoint_url,
    instructions:
      `Set MODEL_URL to this value on the ML API host (Render/Hugging Face Space env vars, ` +
      `see docs/DEPLOY_ML_API.md / docs/DEPLOY_HUGGINGFACE.md) and restart the service.`,
  });
}
