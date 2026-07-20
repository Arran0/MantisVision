import { MeasurementSchemaEditor } from "@/components/admin/MeasurementSchemaEditor";
import { AdminPageHeader } from "@/components/admin/ui";
import { requireAdminPage } from "@/lib/supabase/page-guards";

export default async function SchemaPage() {
  await requireAdminPage();
  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader
        title="Dataset structure"
        subtitle="Edit the measurements the model predicts — species included. Changes apply to labeling right away; they reach the model on the next retrain and go live when that run is promoted."
      />
      <MeasurementSchemaEditor />
    </div>
  );
}
