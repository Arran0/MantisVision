import { RetrainPanel } from "@/components/admin/RetrainPanel";

export default function RetrainPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Retrain</h1>
        <p className="mt-1 text-sm text-slate-600">
          Manually trigger a training run and review it before promoting.
        </p>
      </div>
      <RetrainPanel />
    </div>
  );
}
