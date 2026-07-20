import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { ASSIGNABLE_ROLES, type Role } from "@/lib/roles";
import type { TeamMember } from "@/lib/types";

// Where an invited teammate lands to set their password. Prefer an explicit
// site URL (correct behind proxies / on Vercel); fall back to the request's
// own origin for local dev.
function passwordResetUrl(request: NextRequest): string {
  const base = (process.env.NEXT_PUBLIC_SITE_URL ?? request.nextUrl.origin).replace(/\/$/, "");
  return `${base}/member/password-reset`;
}

// Supabase reports an already-provisioned address a few different ways
// depending on version; match loosely so re-invites fall back to a recovery
// link instead of erroring.
function isAlreadyRegistered(message: string): boolean {
  const m = message.toLowerCase();
  return m.includes("already been registered") || m.includes("already registered") || m.includes("already exists");
}

const PAGE_SIZE = 10;

export async function GET(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const page = Math.max(1, Number(request.nextUrl.searchParams.get("page") ?? "1") || 1);
  const from = (page - 1) * PAGE_SIZE;
  const to = from + PAGE_SIZE - 1;

  const admin = createAdminClient();
  const {
    data,
    error,
    count,
  } = await admin
    .from("profiles")
    .select("id, email, role, created_at", { count: "exact" })
    .order("created_at", { ascending: true })
    .range(from, to);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  const members: TeamMember[] = (data ?? []).map((row) => ({
    id: row.id,
    email: row.email,
    role: (row.role ?? "viewer") as Role,
    createdAt: row.created_at,
    isSelf: row.id === auth.context.userId,
  }));

  const total = count ?? 0;
  const hasMore = page * PAGE_SIZE < total;

  return NextResponse.json({ members, page, total, hasMore });
}

export async function POST(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  const role = body?.role as Role | undefined;

  if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    return NextResponse.json({ error: "Enter a valid email address." }, { status: 400 });
  }
  if (!role || !ASSIGNABLE_ROLES.includes(role as (typeof ASSIGNABLE_ROLES)[number])) {
    return NextResponse.json({ error: "Choose a valid account level." }, { status: 400 });
  }

  const admin = createAdminClient();
  const redirectTo = passwordResetUrl(request);

  // Best effort: send the branded Supabase invite email. This only works if
  // the project has SMTP configured — when it doesn't, this errors and we fall
  // through to the link-only path below.
  let emailed = false;
  let existingUser = false;
  let userId: string | null = null;
  const invited = await admin.auth.admin.inviteUserByEmail(email, { redirectTo });
  if (!invited.error && invited.data?.user) {
    emailed = true;
    userId = invited.data.user.id;
  } else if (invited.error && isAlreadyRegistered(invited.error.message)) {
    existingUser = true;
  }

  // Only mint a separate action link when the branded email didn't already go
  // out. generateLink() issues a brand-new token for the user, which
  // invalidates whichever token is currently outstanding — including the one
  // just emailed above — so calling it unconditionally used to make every
  // emailed invite link die on arrival ("Email link is invalid or expired").
  let actionLink: string | null = null;
  if (!emailed) {
    let { data, error } = await admin.auth.admin.generateLink({
      type: existingUser ? "recovery" : "invite",
      email,
      options: { redirectTo },
    });

    // If we guessed 'invite' but the user turned out to already exist, retry
    // as recovery so the admin still gets a usable link.
    if (error && isAlreadyRegistered(error.message)) {
      existingUser = true;
      ({ data, error } = await admin.auth.admin.generateLink({
        type: "recovery",
        email,
        options: { redirectTo },
      }));
    }

    if (error || !data?.user) {
      return NextResponse.json(
        { error: `Could not create the invite: ${error?.message ?? "unknown error"}` },
        { status: 502 }
      );
    }

    userId = data.user.id;
    actionLink = data.properties?.action_link ?? null;
  }

  if (!userId) {
    return NextResponse.json({ error: "Could not create the invite: no user id returned." }, { status: 502 });
  }

  // Apply the chosen level. The auth trigger seeds the profile as 'viewer';
  // this promotes it to what the admin picked. Email is backfilled too in case
  // the row predates email capture.
  const { error: roleError } = await admin
    .from("profiles")
    .update({ role, email })
    .eq("id", userId);

  if (roleError) {
    return NextResponse.json({ error: `Invite created but role update failed: ${roleError.message}` }, { status: 502 });
  }

  return NextResponse.json({
    email,
    role,
    emailed,
    reused: existingUser,
    actionLink,
  });
}

export async function PATCH(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const userId = typeof body?.userId === "string" ? body.userId : "";
  const role = body?.role as Role | undefined;

  if (!userId) {
    return NextResponse.json({ error: "Missing user." }, { status: 400 });
  }
  if (!role || !["admin", "contributor", "viewer"].includes(role)) {
    return NextResponse.json({ error: "Invalid account level." }, { status: 400 });
  }
  // Guard against an admin locking themselves out of admin controls.
  if (userId === auth.context.userId && role !== "admin") {
    return NextResponse.json({ error: "You can't change your own admin level." }, { status: 400 });
  }

  const admin = createAdminClient();
  const { error } = await admin.from("profiles").update({ role }).eq("id", userId);
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  return NextResponse.json({ userId, role });
}
