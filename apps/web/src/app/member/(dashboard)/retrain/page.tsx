import { RetrainPanel } from "@/components/admin/RetrainPanel";
import { AdminPageHeader } from "@/components/admin/ui";
import { requireAdminPage } from "@/lib/supabase/page-guards";

export default async function RetrainPage() {
  await requireAdminPage();
  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader title="Retrain" subtitle="Manually trigger a training run and review it before promoting." />
      <RetrainPanel />
    </div>
  );
}
