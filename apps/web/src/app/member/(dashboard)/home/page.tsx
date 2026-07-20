import { redirect } from "next/navigation";
import { getDashboardUser } from "@/lib/supabase/page-guards";
import { createAdminClient } from "@/lib/supabase/admin";
import { AdminPageHeader, AdminStat, AdminCard } from "@/components/admin/ui";
import type { Role } from "@/lib/roles";
import { roleLabel } from "@/lib/roles";

// Server-rendered so the counts are computed with the service-role client and
// never leave the server except as plain numbers. Both admins and contributors
// land here; admins get an extra team-composition row.
export default async function HomePage() {
  const user = await getDashboardUser();
  if (!user) redirect("/member/login");

  const admin = createAdminClient();

  const [{ count: myCount }, { count: totalCount }] = await Promise.all([
    admin
      .from("training_images")
      .select("*", { count: "exact", head: true })
      .eq("created_by", user.id),
    admin.from("training_images").select("*", { count: "exact", head: true }),
  ]);

  const mine = myCount ?? 0;
  const total = totalCount ?? 0;
  const share = total > 0 ? Math.round((mine / total) * 100) : 0;

  // Admin-only: how the team breaks down by level.
  let roleCounts: Record<Role, number> | null = null;
  if (user.role === "admin") {
    const { data: profiles } = await admin.from("profiles").select("role");
    const counts: Record<Role, number> = { admin: 0, contributor: 0, viewer: 0 };
    for (const row of profiles ?? []) {
      const role = (row.role ?? "viewer") as Role;
      if (role in counts) counts[role] += 1;
    }
    roleCounts = counts;
  }

  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader
        title="Welcome back"
        subtitle={`Signed in as ${roleLabel(user.role)}${user.email ? ` — ${user.email}` : ""}`}
      />

      <section className="flex flex-col gap-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">Your contributions</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <AdminStat label="Photos you contributed" value={mine} hint="Labeled images you added" />
          <AdminStat label="Total dataset photos" value={total} hint="Across all contributors" />
          <AdminStat
            label="Your share"
            value={`${share}%`}
            hint={total > 0 ? `${mine} of ${total} photos` : "No photos yet"}
          />
        </div>
      </section>

      {roleCounts && (
        <section className="flex flex-col gap-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">Team</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <AdminStat label="Admins" value={roleCounts.admin} />
            <AdminStat label="Contributors" value={roleCounts.contributor} />
            <AdminStat
              label="Total members"
              value={roleCounts.admin + roleCounts.contributor + roleCounts.viewer}
              hint={roleCounts.viewer > 0 ? `${roleCounts.viewer} without a level yet` : undefined}
            />
          </div>
        </section>
      )}

      <AdminCard className="p-5">
        <p className="text-sm text-zinc-600">
          {user.role === "admin"
            ? "Add photos under Dataset, edit Structure, trigger a Retrain, and invite teammates under Team."
            : "Head to Dataset to label photos — every image you add grows the training set."}
        </p>
      </AdminCard>
    </div>
  );
}
