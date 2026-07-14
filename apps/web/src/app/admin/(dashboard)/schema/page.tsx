import { MeasurementSchemaEditor } from "@/components/admin/MeasurementSchemaEditor";
import { AdminPageHeader } from "@/components/admin/ui";

export default function SchemaPage() {
  return (
    <div className="flex flex-col gap-5">
      <AdminPageHeader
        title="Dataset structure"
        subtitle="Edit the species and measurements the model predicts. Changes apply to labeling right away; they reach the model on the next retrain and go live when that run is promoted."
      />
      <MeasurementSchemaEditor />
    </div>
  );
}
