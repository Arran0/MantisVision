"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
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

  const links = LINKS.filter((link) => !link.adminOnly || role === "admin");

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
                active ? "bg-dewberry-600 text-white" : "text-zinc-600 hover:bg-zinc-100"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </div>
      {email && (
        <span className="ml-auto hidden items-center gap-2 text-sm text-zinc-500 sm:flex">
          <span>{email}</span>
          <span className="rounded-sm border border-zinc-300 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-zinc-500">
            {role}
          </span>
        </span>
      )}
    </nav>
  );
}
