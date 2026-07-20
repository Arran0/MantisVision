import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import type { ModelRun } from "@/lib/types";

const WORKFLOW_FILE = "retrain.yml";

function toModelRun(row: Record<string, unknown>): ModelRun {
  return {
    id: row.id as string,
    createdAt: row.created_at as string,
    status: row.status as ModelRun["status"],
    githubRunId: (row.github_run_id as string | null) ?? null,
    datasetImageCount: (row.dataset_image_count as number | null) ?? null,
    metrics: (row.metrics as Record<string, unknown> | null) ?? null,
    checkpointUrl: (row.checkpoint_url as string | null) ?? null,
    error: (row.error as string | null) ?? null,
    promotedAt: (row.promoted_at as string | null) ?? null,
  };
}

// Dispatches .github/workflows/retrain.yml on GitHub Actions rather than
// running training inline: keeps CPU/RAM-hungry training off the always-on
// FastAPI serving container (ml/src/api/main.py), which would otherwise risk
// starving live /predict traffic mid-run.
async function dispatchWorkflow(modelRunId: string): Promise<{ ok: true } | { ok: false; error: string }> {
  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.GITHUB_REPO;
  if (!token || !repo) {
    return { ok: false, error: "GITHUB_TOKEN and GITHUB_REPO must be set to trigger retraining." };
  }

  const response = await fetch(
    `https://api.github.com/repos/${repo}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: "main", inputs: { model_run_id: modelRunId } }),
    }
  );

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    return { ok: false, error: `GitHub Actions dispatch failed (HTTP ${response.status}): ${body}` };
  }
  return { ok: true };
}

export async function POST() {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const admin = createAdminClient();
  const { data: row, error: insertError } = await admin
    .from("model_runs")
    .insert({ triggered_by: auth.context.userId, status: "queued" })
    .select()
    .single();

  if (insertError || !row) {
    return NextResponse.json(
      { error: `Failed to create the run: ${insertError?.message ?? "unknown error"}` },
      { status: 502 }
    );
  }

  const dispatch = await dispatchWorkflow(row.id);
  if (!dispatch.ok) {
    await admin.from("model_runs").update({ status: "failed", error: dispatch.error }).eq("id", row.id);
    return NextResponse.json({ error: dispatch.error }, { status: 502 });
  }

  return NextResponse.json({ run: toModelRun(row) }, { status: 201 });
}

export async function GET(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const admin = createAdminClient();
  const { data: rows, error } = await admin
    .from("model_runs")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(20);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  return NextResponse.json({ runs: (rows ?? []).map(toModelRun) });
}
