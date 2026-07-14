"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

const LINKS = [
  { href: "/admin", label: "Overview" },
  { href: "/admin/dataset", label: "Dataset" },
  { href: "/admin/taxonomy", label: "Structure" },
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
    <nav className="flex flex-wrap items-center gap-4 border-b border-slate-200/70 pb-4">
      <div className="flex gap-2">
        {LINKS.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
              pathname === link.href
                ? "bg-ocean-500/15 text-ocean-700"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            {link.label}
          </Link>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-3 text-sm text-slate-500">
        {email && <span>{email}</span>}
        <button type="button" onClick={signOut} className="mv-btn-orange px-4 py-2 text-sm">
          Sign out
        </button>
      </div>
    </nav>
  );
}
