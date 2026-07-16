import { AdminPageHeader } from "@/components/admin/ui";
import { TeamPanel } from "@/components/admin/TeamPanel";
import { requireAdminPage } from "@/lib/supabase/page-guards";

export default async function TeamPage() {
  await requireAdminPage();
  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader
        title="Team"
        subtitle="Invite people to help label photos, and set who can manage the dataset."
      />
      <TeamPanel />
    </div>
  );
}
