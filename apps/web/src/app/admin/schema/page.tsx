import { MeasurementSchemaEditor } from "@/components/admin/MeasurementSchemaEditor";

export default function SchemaPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Dataset structure</h1>
        <p className="mt-1 text-sm text-slate-600">
          Edit the species and the measurements the model predicts — classifications, regressions, and
          segmentations — including the preset explanation/recommendation copy. Changes apply to labeling and
          validation right away; they reach the model on the next retrain and go live when that run is promoted.
        </p>
      </div>
      <MeasurementSchemaEditor />
    </div>
  );
}
