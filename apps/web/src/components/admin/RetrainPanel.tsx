"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { ModelRun } from "@/lib/types";
import { AdminBadge, AdminButton, AdminCard, sectionHeadingClass } from "@/components/admin/ui";

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
    <div className="flex flex-col gap-5">
      <AdminCard className="flex flex-col gap-3 p-5">
        <h2 className={sectionHeadingClass}>Retrain model</h2>
        <p className="text-sm text-zinc-600">
          Runs training on every currently labeled photo in the dataset, then reports evaluation metrics here.
          Nothing goes live until you explicitly promote a completed run.
        </p>
        <AdminButton type="button" onClick={trigger} disabled={triggering} className="self-start">
          {triggering ? "Starting…" : "Retrain model"}
        </AdminButton>
        {message && <p className="whitespace-pre-wrap text-sm text-zinc-700">{message}</p>}
      </AdminCard>

      <div className="flex flex-col gap-3">
        {runs.length === 0 && <p className="text-sm text-zinc-500">No retraining runs yet.</p>}
        <AnimatePresence initial={false}>
          {runs.map((run) => (
            <motion.div key={run.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15 }}>
              <AdminCard className="flex flex-col gap-2 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-zinc-800">{new Date(run.createdAt).toLocaleString()}</span>
                  <StatusBadge status={run.status} />
                </div>
                {run.datasetImageCount !== null && <p className="text-sm text-zinc-600">{run.datasetImageCount} images used</p>}
                {run.metrics && (
                  <pre className="max-h-48 overflow-auto rounded-lg bg-zinc-50 p-3 text-xs text-zinc-700">
                    {JSON.stringify(run.metrics, null, 2)}
                  </pre>
                )}
                {run.error && <p className="text-sm text-rose-600">{run.error}</p>}
                {run.status === "completed" && !run.promotedAt && (
                  <AdminButton type="button" onClick={() => promote(run.id)} disabled={promotingId === run.id} className="self-start">
                    {promotingId === run.id ? "Promoting…" : "Promote to production"}
                  </AdminButton>
                )}
                {run.promotedAt && (
                  <p className="text-sm text-seaweed-600">Promoted {new Date(run.promotedAt).toLocaleString()}</p>
                )}
              </AdminCard>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: ModelRun["status"] }) {
  const tones: Record<ModelRun["status"], "zinc" | "ocean" | "seaweed" | "rose"> = {
    queued: "zinc",
    running: "ocean",
    completed: "seaweed",
    failed: "rose",
  };
  return <AdminBadge tone={tones[status]}>{status}</AdminBadge>;
}
