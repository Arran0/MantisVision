"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

const LINKS = [
  { href: "/admin/dataset", label: "Dataset" },
  { href: "/admin/schema", label: "Structure" },
  { href: "/admin/retrain", label: "Retrain" },
];

export function AdminNav({ email }: { email: string | null }) {
  const pathname = usePathname();
  const router = useRouter();

  async function signOut() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/admin/login");
    router.refresh();
  }

  return (
    <nav className="flex flex-wrap items-center gap-4 border-b border-zinc-200 pb-3">
      <div className="flex gap-1">
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                active ? "bg-zinc-900 text-white" : "text-zinc-600 hover:bg-zinc-100"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
      <div className="ml-auto flex items-center gap-3 text-sm text-zinc-500">
        {email && <span className="hidden sm:inline">{email}</span>}
        <button
          type="button"
          onClick={signOut}
          className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100"
        >
          Sign out
        </button>
      </div>
    </nav>
  );
}
