import type { TrainingImage } from "@/lib/types";
import { AdminCard } from "@/components/admin/ui";

export function DatasetTable({ images, loading }: { images: TrainingImage[]; loading: boolean }) {
  if (loading) {
    return <p className="text-sm text-zinc-500">Loading…</p>;
  }
  if (images.length === 0) {
    return <p className="text-sm text-zinc-500">No labeled photos yet.</p>;
  }

  return (
    <AdminCard className="overflow-hidden">
      <div className="flex bg-zinc-900 px-4 py-2.5">
        <span className="w-14 flex-shrink-0" />
        <span className="flex-1 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Species</span>
        <span className="hidden flex-[2] text-[10px] font-bold uppercase tracking-widest text-zinc-400 sm:block">
          Measurements
        </span>
        <span className="w-24 flex-shrink-0 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Status</span>
        <span className="hidden w-28 flex-shrink-0 text-right text-[10px] font-bold uppercase tracking-widest text-zinc-400 sm:block">
          Added
        </span>
      </div>
      <div className="max-h-[32rem] overflow-y-auto">
        {images.map((image) => (
          <div key={image.id} className="flex items-center gap-0 border-b border-zinc-100 px-4 py-2.5 last:border-0 hover:bg-zinc-50">
            <div className="w-14 flex-shrink-0">
              {image.thumbnailUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={image.thumbnailUrl} alt="" className="h-10 w-10 rounded-lg object-cover" />
              ) : (
                <div className="h-10 w-10 rounded-lg bg-zinc-100" />
              )}
            </div>
            <span className="flex-1 truncate text-sm italic text-zinc-800">{image.species ?? "—"}</span>
            <span className="hidden flex-[2] truncate text-xs text-zinc-600 sm:block">
              {Object.keys(image.measurements).length === 0
                ? "—"
                : Object.entries(image.measurements)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(" · ")}
            </span>
            <span className="w-24 flex-shrink-0 text-xs text-zinc-600">{image.status}</span>
            <span className="hidden w-28 flex-shrink-0 text-right text-xs text-zinc-400 sm:block">
              {new Date(image.createdAt).toLocaleDateString()}
            </span>
          </div>
        ))}
      </div>
    </AdminCard>
  );
}
