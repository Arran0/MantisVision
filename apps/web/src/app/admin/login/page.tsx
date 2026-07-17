"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { AdminButton, AdminCard, AdminField, AdminInput } from "@/components/admin/ui";

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

    router.push("/admin/home");
    router.refresh();
  }

  return (
    <div className="min-h-screen bg-zinc-50">
      <main className="mx-auto flex max-w-sm flex-col gap-6 px-5 py-16 sm:px-8">
        <h1 className="text-xl font-bold tracking-tight text-zinc-900">Admin sign in</h1>
        <AdminCard className="p-6">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <AdminField label="Email">
              <AdminInput
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </AdminField>
            <AdminField label="Password">
              <AdminInput
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </AdminField>
            {error && <p className="text-sm text-rose-600">{error}</p>}
            <AdminButton type="submit" disabled={loading} className="w-full">
              {loading ? "Signing in…" : "Sign in"}
            </AdminButton>
          </form>
          <p className="mt-5 text-center text-xs text-zinc-400">This sign-in is for admin use only.</p>
        </AdminCard>
      </main>
    </div>
  );
}
