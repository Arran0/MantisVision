import { RetrainPanel } from "@/components/admin/RetrainPanel";
import { AdminPageHeader } from "@/components/admin/ui";

export default function RetrainPage() {
  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader title="Retrain" subtitle="Manually trigger a training run and review it before promoting." />
      <RetrainPanel />
    </div>
  );
}
