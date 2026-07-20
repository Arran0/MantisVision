"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { ModelRun } from "@/lib/types";
import { AdminBadge, AdminButton, AdminCard, sectionHeadingClass } from "@/components/admin/ui";

const POLL_MS = 5_000;
const PAGE_SIZE = 10;

export function RetrainPanel() {
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [promotingId, setPromotingId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Refetches the currently-visible window (grows by PAGE_SIZE on "Load
  // more") so a poll or an action refreshes every run already on screen, not
  // just the newest page.
  const refresh = useCallback(async (windowSize: number) => {
    const response = await fetch(`/api/member/retrain?limit=${windowSize}`);
    const payload = await response.json().catch(() => null);
    setRuns(payload?.runs ?? []);
    setTotal(payload?.total ?? 0);
    setHasMore(Boolean(payload?.hasMore));
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        await refresh(PAGE_SIZE);
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh]);

  async function loadMore() {
    const nextLimit = limit + PAGE_SIZE;
    setLoadingMore(true);
    try {
      await refresh(nextLimit);
      setLimit(nextLimit);
    } finally {
      setLoadingMore(false);
    }
  }

  // Poll while any run is in flight; stop as soon as none are.
  useEffect(() => {
    const hasActiveRun = runs.some((run) => run.status === "queued" || run.status === "running");
    if (hasActiveRun && !pollRef.current) {
      pollRef.current = setInterval(() => refresh(limit), POLL_MS);
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
  }, [runs, refresh, limit]);

  async function trigger() {
    setTriggering(true);
    setMessage(null);
    try {
      const response = await fetch("/api/member/retrain", { method: "POST" });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? "Failed to trigger retraining.");
      await refresh(limit);
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
      const response = await fetch("/api/member/retrain/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelRunId: runId }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? "Failed to promote.");
      setMessage(payload?.message ?? "Model promoted — it is now serving live predictions.");
      await refresh(limit);
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
        {loading ? (
          <p className="text-sm text-zinc-500">Loading…</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-zinc-500">No retraining runs yet.</p>
        ) : (
          <div className="max-h-[32rem] overflow-y-auto pr-1">
            <AnimatePresence initial={false}>
              <div className="flex flex-col gap-3">
                {runs.map((run) => (
                  <motion.div key={run.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15 }}>
                    <AdminCard className="flex flex-col gap-2 p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold text-zinc-800">{new Date(run.createdAt).toLocaleString()}</span>
                        <StatusBadge status={run.status} />
                      </div>
                      {run.datasetImageCount !== null && <p className="text-sm text-zinc-600">{run.datasetImageCount} images used</p>}
                      {run.metrics && (
                        <pre className="max-h-48 overflow-auto rounded-sm bg-zinc-50 p-3 text-xs text-zinc-700">
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
              </div>
            </AnimatePresence>
          </div>
        )}

        {!loading && runs.length > 0 && (
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs text-zinc-500">
              Showing {runs.length}
              {total > 0 ? ` of ${total}` : ""} run{runs.length === 1 ? "" : "s"}
            </p>
            {hasMore && (
              <AdminButton type="button" variant="secondary" onClick={loadMore} disabled={loadingMore}>
                {loadingMore ? "Loading…" : "Load more"}
              </AdminButton>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: ModelRun["status"] }) {
  const tones: Record<ModelRun["status"], "zinc" | "dewberry" | "seaweed" | "rose"> = {
    queued: "zinc",
    running: "dewberry",
    completed: "seaweed",
    failed: "rose",
  };
  return <AdminBadge tone={tones[status]}>{status}</AdminBadge>;
}
