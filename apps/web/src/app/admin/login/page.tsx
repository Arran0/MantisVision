"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export default function AdminLoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setLoading(true);

    const supabase = createClient();
    const { error: signInError } = await supabase.auth.signInWithPassword({ email, password });

    if (signInError) {
      setError(signInError.message);
      setLoading(false);
      return;
    }

    router.push("/admin");
    router.refresh();
  }

  return (
    <main className="mx-auto flex max-w-sm flex-col gap-6 px-5 py-16 sm:px-8">
      <h1 className="text-2xl font-bold tracking-tight text-slate-900">Admin sign in</h1>
      <form onSubmit={handleSubmit} className="mv-card flex flex-col gap-4 p-6">
        <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
          Email
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
          Password
          <input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="rounded-lg border border-slate-300 bg-white/80 px-3 py-2 text-base text-slate-900"
          />
        </label>
        {error && <p className="text-sm text-coral-600">{error}</p>}
        <button type="submit" disabled={loading} className="mv-btn-blue w-full">
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
