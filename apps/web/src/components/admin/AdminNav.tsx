"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import type { Role } from "@/lib/roles";

type NavLink = { href: string; label: string; adminOnly?: boolean };

const LINKS: NavLink[] = [
  { href: "/admin/home", label: "Home" },
  { href: "/admin/dataset", label: "Dataset" },
  { href: "/admin/schema", label: "Structure", adminOnly: true },
  { href: "/admin/retrain", label: "Retrain", adminOnly: true },
  { href: "/admin/team", label: "Team", adminOnly: true },
];

export function AdminNav({ email, role }: { email: string | null; role: Role }) {
  const pathname = usePathname();
  const router = useRouter();

  const links = LINKS.filter((link) => !link.adminOnly || role === "admin");

  async function signOut() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/admin/login");
    router.refresh();
  }

  return (
    <nav className="flex flex-wrap items-center gap-4 border-b border-zinc-200 pb-3">
      <div className="flex flex-wrap gap-1">
        {links.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-sm px-3 py-1.5 text-sm font-medium transition-colors ${
                active ? "bg-zinc-900 text-white" : "text-zinc-600 hover:bg-zinc-100"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
      <div className="ml-auto flex items-center gap-3 text-sm text-zinc-500">
        {email && (
          <span className="hidden items-center gap-2 sm:flex">
            <span>{email}</span>
            <span className="rounded-sm border border-zinc-300 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-zinc-500">
              {role}
            </span>
          </span>
        )}
        <button
          type="button"
          onClick={signOut}
          className="rounded-sm border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100"
        >
          Sign out
        </button>
      </div>
    </nav>
  );
}
