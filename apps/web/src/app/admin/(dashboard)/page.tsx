import Link from "next/link";

export default function AdminOverviewPage() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Link href="/admin/dataset" className="mv-card block p-6 transition hover:-translate-y-0.5">
        <h2 className="text-lg font-semibold text-slate-900">Dataset</h2>
        <p className="mt-1 text-sm text-slate-600">
          Upload photos and label species, colour, health, and other metadata to grow the
          training dataset.
        </p>
      </Link>
      <Link href="/admin/retrain" className="mv-card block p-6 transition hover:-translate-y-0.5">
        <h2 className="text-lg font-semibold text-slate-900">Retrain</h2>
        <p className="mt-1 text-sm text-slate-600">
          Trigger a training run on the current dataset, review its metrics, and promote a new
          checkpoint to production.
        </p>
      </Link>
    </div>
  );
}
