"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { AdminButton, AdminCard, AdminField, AdminInput } from "@/components/admin/ui";

type Phase = "verifying" | "ready" | "invalid" | "done";

// Where an invited teammate lands from their invite/recovery link to choose a
// password. The link carries a token — usually in the URL hash (implicit
// flow), sometimes as ?code= or ?token_hash=&type= — which we exchange for a
// session here, then let them set a password with supabase.auth.updateUser().
//
// This page sits OUTSIDE the (dashboard) route group and is excluded from the
// middleware auth gate, because the token lives in the URL hash the server
// never sees: it must render for a signed-out visitor so the client can finish
// establishing the session.
export default function PasswordResetPage() {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>("verifying");
  const [errorDetail, setErrorDetail] = useState<string | null>(null);

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const supabase = createClient();

    // Strip the token from the address bar once we've read it, so it isn't
    // left in history or copied by accident.
    function scrubUrl() {
      if (typeof window !== "undefined") {
        window.history.replaceState(null, "", window.location.pathname);
      }
    }

    (async () => {
      const hash = typeof window !== "undefined" ? window.location.hash.replace(/^#/, "") : "";
      const hashParams = new URLSearchParams(hash);
      const query = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : new URLSearchParams();

      // Supabase reports a bad/expired link via error params in the hash or query.
      const errorDescription = hashParams.get("error_description") ?? query.get("error_description");
      if (errorDescription) {
        if (!cancelled) {
          setErrorDetail(errorDescription.replace(/\+/g, " "));
          setPhase("invalid");
        }
        scrubUrl();
        return;
      }

      try {
        const accessToken = hashParams.get("access_token");
        const refreshToken = hashParams.get("refresh_token");
        const code = query.get("code");
        const tokenHash = query.get("token_hash");
        const type = query.get("type");

        if (accessToken && refreshToken) {
          const { error } = await supabase.auth.setSession({
            access_token: accessToken,
            refresh_token: refreshToken,
          });
          if (error) throw error;
        } else if (code) {
          const { error } = await supabase.auth.exchangeCodeForSession(code);
          if (error) throw error;
        } else if (tokenHash && type) {
          const { error } = await supabase.auth.verifyOtp({
            token_hash: tokenHash,
            type: type as "invite" | "recovery" | "email",
          });
          if (error) throw error;
        }
        scrubUrl();

        // Whichever path ran (or if the client auto-detected the session from
        // the URL), confirm we actually have one before showing the form.
        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (cancelled) return;
        setPhase(session ? "ready" : "invalid");
      } catch (err) {
        scrubUrl();
        if (cancelled) return;
        setErrorDetail(err instanceof Error ? err.message : "This link could not be verified.");
        setPhase("invalid");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setFormError(null);

    if (password.length < 8) {
      setFormError("Use at least 8 characters.");
      return;
    }
    if (password !== confirm) {
      setFormError("Passwords don't match.");
      return;
    }

    setSaving(true);
    const supabase = createClient();
    const { error } = await supabase.auth.updateUser({ password });
    if (error) {
      setFormError(error.message);
      setSaving(false);
      return;
    }

    setPhase("done");
    // The session is already active — send them straight into the dashboard.
    setTimeout(() => {
      router.push("/member/home");
      router.refresh();
    }, 900);
  }

  return (
    <div className="min-h-screen bg-zinc-50">
      <main className="mx-auto flex max-w-sm flex-col gap-6 px-5 py-16 sm:px-8">
        <h1 className="text-xl font-bold tracking-tight text-zinc-900">Set your password</h1>

        {phase === "verifying" && (
          <AdminCard className="p-6">
            <p className="text-sm text-zinc-500">Verifying your invite…</p>
          </AdminCard>
        )}

        {phase === "invalid" && (
          <AdminCard className="p-6">
            <p className="text-sm text-zinc-700">
              This invite link is invalid or has expired. Ask an admin to send you a new one.
            </p>
            {errorDetail && <p className="mt-2 text-xs text-zinc-400">{errorDetail}</p>}
            <AdminButton
              type="button"
              variant="secondary"
              className="mt-4"
              onClick={() => router.push("/member/login")}
            >
              Go to sign in
            </AdminButton>
          </AdminCard>
        )}

        {phase === "ready" && (
          <AdminCard className="p-6">
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <p className="text-sm text-zinc-600">Choose a password to finish setting up your account.</p>
              <AdminField label="New password">
                <AdminInput
                  type="password"
                  autoComplete="new-password"
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </AdminField>
              <AdminField label="Confirm password">
                <AdminInput
                  type="password"
                  autoComplete="new-password"
                  required
                  value={confirm}
                  onChange={(event) => setConfirm(event.target.value)}
                />
              </AdminField>
              {formError && <p className="text-sm text-rose-600">{formError}</p>}
              <AdminButton type="submit" disabled={saving} className="w-full">
                {saving ? "Saving…" : "Set password & sign in"}
              </AdminButton>
            </form>
          </AdminCard>
        )}

        {phase === "done" && (
          <AdminCard className="p-6">
            <p className="text-sm text-zinc-700">Password set. Taking you to the dashboard…</p>
          </AdminCard>
        )}
      </main>
    </div>
  );
}
