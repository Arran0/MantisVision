"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ModelRun } from "@/lib/types";

const POLL_MS = 5_000;

export function RetrainPanel() {
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [triggering, setTriggering] = useState(false);
  const [promotingId, setPromotingId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    const response = await fetch("/api/admin/retrain");
    const payload = await response.json().catch(() => null);
    setRuns(payload?.runs ?? []);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while any run is in flight; stop as soon as none are.
  useEffect(() => {
    const hasActiveRun = runs.some((run) => run.status === "queued" || run.status === "running");
    if (hasActiveRun && !pollRef.current) {
      pollRef.current = setInterval(refresh, POLL_MS);
    } else if (!hasActiveRun && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [runs, refresh]);

  async function trigger() {
    setTriggering(true);
    setMessage(null);
    try {
      const response = await fetch("/api/admin/retrain", { method: "POST" });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? "Failed to trigger retraining.");
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setTriggering(false);
    }
  }

  async function promote(runId: string) {
    setPromotingId(runId);
    setMessage(null);
    try {
      const response = await fetch("/api/admin/retrain/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelRunId: runId }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? "Failed to promote.");
      setMessage(payload?.message ?? "Model promoted — it is now serving live predictions.");
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setPromotingId(null);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="mv-card flex flex-col gap-3 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Retrain model</h2>
        <p className="text-sm text-slate-600">
          Runs training on every currently labeled photo in the dataset, then reports evaluation
          metrics here. Nothing goes live until you explicitly promote a completed run.
        </p>
        <button type="button" onClick={trigger} disabled={triggering} className="mv-btn-blue self-start">
          {triggering ? "Starting…" : "Retrain model"}
        </button>
        {message && <p className="whitespace-pre-wrap text-sm text-slate-700">{message}</p>}
      </div>

      <div className="flex flex-col gap-3">
        {runs.length === 0 && <p className="text-sm text-slate-500">No retraining runs yet.</p>}
        {runs.map((run) => (
          <div key={run.id} className="mv-card flex flex-col gap-2 p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-800">
                {new Date(run.createdAt).toLocaleString()}
              </span>
              <StatusBadge status={run.status} />
            </div>
            {run.datasetImageCount !== null && (
              <p className="text-sm text-slate-600">{run.datasetImageCount} images used</p>
            )}
            {run.metrics && (
              <pre className="max-h-48 overflow-auto rounded-lg bg-slate-50 p-3 text-xs text-slate-700">
                {JSON.stringify(run.metrics, null, 2)}
              </pre>
            )}
            {run.error && <p className="text-sm text-coral-600">{run.error}</p>}
            {run.status === "completed" && !run.promotedAt && (
              <button
                type="button"
                onClick={() => promote(run.id)}
                disabled={promotingId === run.id}
                className="mv-btn-orange self-start"
              >
                {promotingId === run.id ? "Promoting…" : "Promote to production"}
              </button>
            )}
            {run.promotedAt && (
              <p className="text-sm text-seaweed-600">
                Promoted {new Date(run.promotedAt).toLocaleString()}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: ModelRun["status"] }) {
  const styles: Record<ModelRun["status"], string> = {
    queued: "bg-slate-100 text-slate-700",
    running: "bg-ocean-100 text-ocean-700",
    completed: "bg-seaweed-100 text-seaweed-700",
    failed: "bg-coral-500/15 text-coral-600",
  };
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-semibold ${styles[status]}`}>{status}</span>
  );
}
