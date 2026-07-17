import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { ML_API_URL } from "@/lib/config";

// The ML API's own checkpoint download timeout is 60s (CHECKPOINT_DOWNLOAD_TIMEOUT_S
// in ml/src/api/main.py), plus time to build the Predictor — well past Vercel's
// default function timeout, which would otherwise kill this route and return a
// non-JSON timeout page before the reload finishes.
export const maxDuration = 60;

// Promotion hot-swaps the live model with zero manual steps. The mechanics are
// automatic, but the *decision* stays a deliberate admin action (an admin
// reviews the run's metrics, then clicks Promote) — a bad retrain still can't
// reach production without a human approving it.
//
// Order matters: we call the ML API's /admin/reload FIRST and only record
// promoted_at/promoted_by in Supabase if that swap succeeds. This keeps
// model_runs honest — nothing is marked "promoted" unless it is actually
// serving live traffic.
export async function POST(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const modelRunId = body?.modelRunId;
  if (typeof modelRunId !== "string") {
    return NextResponse.json({ error: "'modelRunId' is required." }, { status: 400 });
  }

  const reloadToken = process.env.ML_API_ADMIN_TOKEN;
  if (!reloadToken) {
    return NextResponse.json(
      {
        error:
          "ML_API_ADMIN_TOKEN is not set on the web app, so the model can't be reloaded. " +
          "Set it (matching RELOAD_TOKEN on the ML API host) and try again.",
      },
      { status: 500 }
    );
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

  // 1. Hot-swap the live model. If this fails, we do NOT touch model_runs.
  let reload: Response;
  try {
    reload = await fetch(`${ML_API_URL}/admin/reload`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${reloadToken}`,
      },
      body: JSON.stringify({ model_url: run.checkpoint_url }),
    });
  } catch {
    return NextResponse.json(
      { error: `Could not reach the ML API at ${ML_API_URL} to reload the model.` },
      { status: 502 }
    );
  }

  const reloadPayload = await reload.json().catch(() => null);
  if (!reload.ok) {
    return NextResponse.json(
      {
        error:
          reloadPayload?.detail ??
          reloadPayload?.error ??
          `ML API reload failed (HTTP ${reload.status}). The previous model is still serving.`,
      },
      { status: 502 }
    );
  }

  // 2. Model is live — now record the promotion.
  const { error: updateError } = await admin
    .from("model_runs")
    .update({ promoted_at: new Date().toISOString(), promoted_by: auth.context.userId })
    .eq("id", modelRunId);

  if (updateError) {
    // The swap already happened; surface that the record didn't update rather
    // than pretending nothing changed.
    return NextResponse.json(
      {
        error:
          `The model was reloaded and is live, but recording the promotion failed: ` +
          `${updateError.message}. Retry to reconcile the record.`,
      },
      { status: 502 }
    );
  }

  return NextResponse.json({
    ok: true,
    message: "Model promoted — it is now serving live predictions.",
    speciesClasses: reloadPayload?.species_classes ?? null,
    checkpointUrl: run.checkpoint_url,
  });
}
