"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AdminBadge,
  AdminButton,
  AdminCard,
  AdminField,
  AdminInput,
  AdminSelect,
  sectionHeadingClass,
} from "@/components/admin/ui";
import { ASSIGNABLE_ROLES, roleLabel, type Role } from "@/lib/roles";
import type { TeamMember } from "@/lib/types";

type InviteResult = { email: string; role: Role; emailed: boolean; reused: boolean; actionLink: string | null };

function roleTone(role: Role): "dewberry" | "seaweed" | "zinc" {
  if (role === "admin") return "dewberry";
  if (role === "contributor") return "seaweed";
  return "zinc";
}

export function TeamPanel() {
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);

  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("contributor");
  const [inviting, setInviting] = useState(false);
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [invite, setInvite] = useState<InviteResult | null>(null);
  const [copied, setCopied] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/admin/team");
      const payload = await response.json().catch(() => null);
      setMembers(payload?.members ?? []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleInvite(event: React.FormEvent) {
    event.preventDefault();
    setInviting(true);
    setInviteError(null);
    setInvite(null);
    setCopied(false);
    try {
      const response = await fetch("/api/admin/team", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, role }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.error ?? `Invite failed (HTTP ${response.status}).`);
      setInvite(payload as InviteResult);
      setEmail("");
      await refresh();
    } catch (err) {
      setInviteError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setInviting(false);
    }
  }

  async function changeRole(userId: string, nextRole: Role) {
    setMembers((prev) => prev.map((m) => (m.id === userId ? { ...m, role: nextRole } : m)));
    const response = await fetch("/api/admin/team", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId, role: nextRole }),
    });
    if (!response.ok) {
      // Revert on failure so the UI doesn't lie about the saved level.
      await refresh();
    }
  }

  async function copyLink() {
    if (!invite?.actionLink) return;
    try {
      await navigator.clipboard.writeText(invite.actionLink);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <AdminCard className="p-5">
        <form onSubmit={handleInvite} className="flex flex-col gap-4">
          <h2 className={sectionHeadingClass}>Invite a teammate</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-[1fr_12rem]">
            <AdminField label="Email">
              <AdminInput
                type="email"
                required
                placeholder="name@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </AdminField>
            <AdminField label="Account level">
              <AdminSelect value={role} onChange={(event) => setRole(event.target.value as Role)}>
                {ASSIGNABLE_ROLES.map((r) => (
                  <option key={r} value={r}>
                    {roleLabel(r)}
                  </option>
                ))}
              </AdminSelect>
            </AdminField>
          </div>

          {inviteError && <p className="text-sm text-rose-600">{inviteError}</p>}

          <AdminButton type="submit" disabled={inviting} className="self-start">
            {inviting ? "Creating invite…" : "Create invite"}
          </AdminButton>
        </form>

        {invite && (
          <div className="mt-4 flex flex-col gap-2 border border-zinc-200 bg-zinc-50 p-4">
            <p className="text-sm text-zinc-700">
              {invite.reused
                ? `${invite.email} is already in the system — here's a fresh link so they can reset their password.`
                : `Invited ${invite.email} as ${roleLabel(invite.role)}.`}{" "}
              {invite.emailed
                ? "We emailed them an invite. You can also copy this link and send it directly — it lets them set a password and sign in."
                : "Copy this link and send it to them — it lets them set a password and sign in."}
            </p>
            {invite.actionLink ? (
              <div className="flex flex-col gap-2 sm:flex-row">
                <input
                  readOnly
                  value={invite.actionLink}
                  onFocus={(event) => event.target.select()}
                  className="w-full rounded-sm border border-zinc-300 bg-white px-3 py-2 font-mono text-xs text-zinc-700"
                />
                <AdminButton type="button" variant="secondary" onClick={copyLink} className="shrink-0">
                  {copied ? "Copied" : "Copy link"}
                </AdminButton>
              </div>
            ) : (
              <p className="text-xs text-zinc-500">No link was returned — check the Supabase auth configuration.</p>
            )}
          </div>
        )}
      </AdminCard>

      <AdminCard className="overflow-hidden">
        <div className="flex bg-dewberry-900 px-4 py-2.5">
          <span className="flex-1 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Member</span>
          <span className="w-40 flex-shrink-0 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Level</span>
          <span className="hidden w-28 flex-shrink-0 text-right text-[10px] font-bold uppercase tracking-widest text-zinc-400 sm:block">
            Added
          </span>
        </div>
        {loading ? (
          <p className="px-4 py-3 text-sm text-zinc-500">Loading…</p>
        ) : members.length === 0 ? (
          <p className="px-4 py-3 text-sm text-zinc-500">No members yet.</p>
        ) : (
          members.map((member) => (
            <div
              key={member.id}
              className="flex items-center border-b border-zinc-100 px-4 py-2.5 last:border-0 hover:bg-zinc-50"
            >
              <span className="flex flex-1 items-center gap-2 truncate text-sm text-zinc-800">
                <span className="truncate">{member.email ?? "—"}</span>
                {member.isSelf && <AdminBadge tone="amber">You</AdminBadge>}
              </span>
              <span className="w-40 flex-shrink-0">
                {member.isSelf ? (
                  <AdminBadge tone={roleTone(member.role)}>{roleLabel(member.role)}</AdminBadge>
                ) : (
                  <AdminSelect
                    value={member.role}
                    onChange={(event) => changeRole(member.id, event.target.value as Role)}
                    className="max-w-[9rem] py-1.5 text-xs"
                  >
                    <option value="admin">Admin</option>
                    <option value="contributor">Contributor</option>
                    <option value="viewer">No access</option>
                  </AdminSelect>
                )}
              </span>
              <span className="hidden w-28 flex-shrink-0 text-right text-xs text-zinc-400 sm:block">
                {new Date(member.createdAt).toLocaleDateString()}
              </span>
            </div>
          ))
        )}
      </AdminCard>
    </div>
  );
}
