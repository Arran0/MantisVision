import { TaxonomyEditor } from "@/components/admin/TaxonomyEditor";

export default function TaxonomyPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Dataset structure</h1>
        <p className="mt-1 text-sm text-slate-600">
          Edit the species, condition buckets, severities, disease subtypes, the heuristic training anchors,
          and the preset explanation/recommendation copy. Changes apply to labeling and validation right away;
          they reach the model on the next retrain and go live when that run is promoted.
        </p>
      </div>
      <TaxonomyEditor />
    </div>
  );
}
