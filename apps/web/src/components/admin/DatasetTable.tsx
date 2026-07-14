import type { TrainingImage } from "@/lib/types";

export function DatasetTable({ images, loading }: { images: TrainingImage[]; loading: boolean }) {
  if (loading) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }
  if (images.length === 0) {
    return <p className="text-sm text-slate-500">No labeled photos yet.</p>;
  }

  return (
    <div className="mv-card overflow-x-auto p-4">
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs font-medium uppercase tracking-wide text-slate-400">
            <th className="px-3 py-2">Photo</th>
            <th className="px-3 py-2">Species</th>
            <th className="px-3 py-2">Measurements</th>
            <th className="px-3 py-2">Farm</th>
            <th className="px-3 py-2">Status</th>
            <th className="px-3 py-2">Added</th>
          </tr>
        </thead>
        <tbody>
          {images.map((image) => (
            <tr key={image.id} className="border-b border-slate-100 last:border-0">
              <td className="px-3 py-2">
                {image.thumbnailUrl ? (
                  <img
                    src={image.thumbnailUrl}
                    alt=""
                    className="h-12 w-12 rounded-lg object-cover"
                  />
                ) : (
                  <div className="h-12 w-12 rounded-lg bg-slate-100" />
                )}
              </td>
              <td className="px-3 py-2 italic text-slate-800">{image.species ?? "—"}</td>
              <td className="px-3 py-2 text-slate-600">
                {Object.keys(image.measurements).length === 0
                  ? "—"
                  : Object.entries(image.measurements)
                      .map(([key, value]) => `${key}: ${value}`)
                      .join(" · ")}
              </td>
              <td className="px-3 py-2 text-slate-600">{image.farm ?? "—"}</td>
              <td className="px-3 py-2 text-slate-600">{image.status}</td>
              <td className="px-3 py-2 text-slate-500">
                {new Date(image.createdAt).toLocaleDateString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
