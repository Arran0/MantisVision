import { MeasurementSchemaEditor } from "@/components/admin/MeasurementSchemaEditor";
import { AdminPageHeader } from "@/components/admin/ui";
import { requireAdminPage } from "@/lib/supabase/page-guards";

export default async function SchemaPage() {
  await requireAdminPage();
  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader
        title="Dataset structure"
        subtitle="Edit what the model predicts. Changes apply to labeling right away, and reach the model on the next retrain + promote."
      />
      <MeasurementSchemaEditor />
    </div>
  );
}
