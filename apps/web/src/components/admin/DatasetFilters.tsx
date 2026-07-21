"use client";

import { useState, type ReactNode } from "react";
import { classNames, type SchemaDoc } from "@/lib/schema";
import { AdminButton, AdminCard, AdminInput, labelClass } from "@/components/admin/ui";

// One entry per classification measurement's selected classes (OR within a
// measurement, AND across measurements), one per regression measurement's
// [min, max] band, and an optional split selection ("train"/"validation"/
// "test"/"auto" — "auto" means split is null, i.e. not yet assigned).
export interface DatasetFilterState {
  classValues: Record<string, string[]>;
  ranges: Record<string, { min?: number; max?: number }>;
  splits: string[];
}

export const EMPTY_FILTERS: DatasetFilterState = { classValues: {}, ranges: {}, splits: [] };

// Serializes only the non-empty parts of the filter state into the single
// `filters` query param the dataset API expects; null means "no filters" so
// the caller can fall back to the plain paginated query.
export function buildFiltersParam(filters: DatasetFilterState): string | null {
  const classValues = Object.fromEntries(Object.entries(filters.classValues).filter(([, v]) => v.length > 0));
  const ranges = Object.fromEntries(
    Object.entries(filters.ranges).filter(([, r]) => r.min !== undefined || r.max !== undefined)
  );
  const splits = filters.splits;
  if (Object.keys(classValues).length === 0 && Object.keys(ranges).length === 0 && splits.length === 0) return null;
  return JSON.stringify({ classValues, ranges, splits });
}

function activeFilterCount(filters: DatasetFilterState): number {
  return (
    Object.values(filters.classValues).filter((v) => v.length > 0).length +
    Object.values(filters.ranges).filter((r) => r.min !== undefined || r.max !== undefined).length +
    (filters.splits.length > 0 ? 1 : 0)
  );
}

const SPLIT_OPTIONS: { value: string; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "train", label: "Train" },
  { value: "validation", label: "Validation" },
  { value: "test", label: "Test" },
];

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-sm border px-2 py-1 text-[11px] font-medium transition-colors ${
        active ? "border-dewberry-700 bg-dewberry-700 text-white" : "border-zinc-300 text-zinc-600 hover:bg-zinc-100"
      }`}
    >
      {children}
    </button>
  );
}

// Multi-filter panel for the Dataset page: pick one or more classes per
// classification measurement, a value range per regression measurement, and/
// or a split, to narrow the thumbnail list down to that subset. Every filter
// is schema-driven (one control per schema.measurements entry) — no
// hardcoded field list, same as EditImageModal.
export function DatasetFilters({
  schema,
  value,
  onChange,
}: {
  schema: SchemaDoc;
  value: DatasetFilterState;
  onChange: (next: DatasetFilterState) => void;
}) {
  const [open, setOpen] = useState(false);
  const count = activeFilterCount(value);

  function toggleClassValue(key: string, className: string) {
    const current = value.classValues[key] ?? [];
    const next = current.includes(className) ? current.filter((c) => c !== className) : [...current, className];
    onChange({ ...value, classValues: { ...value.classValues, [key]: next } });
  }

  function setRange(key: string, bound: "min" | "max", raw: string) {
    const current = value.ranges[key] ?? {};
    const parsed = raw.trim() === "" ? undefined : Number(raw);
    const next = { ...current, [bound]: parsed !== undefined && Number.isFinite(parsed) ? parsed : undefined };
    onChange({ ...value, ranges: { ...value.ranges, [key]: next } });
  }

  function toggleSplit(split: string) {
    const next = value.splits.includes(split) ? value.splits.filter((s) => s !== split) : [...value.splits, split];
    onChange({ ...value, splits: next });
  }

  const classificationMeasurements = schema.measurements.filter((m) => m.type === "classification");
  const regressionMeasurements = schema.measurements.filter((m) => m.type === "regression");

  return (
    <AdminCard className="p-4">
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-center justify-between text-left">
        <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">
          Filters{count > 0 ? ` (${count} active)` : ""}
        </span>
        <span className="text-xs font-medium text-dewberry-700">{open ? "Hide" : "Show"}</span>
      </button>

      {open && (
        <div className="mt-4 flex flex-col gap-4">
          <div>
            <span className={labelClass}>Dataset split</span>
            <div className="flex flex-wrap gap-1.5">
              {SPLIT_OPTIONS.map((opt) => (
                <Chip key={opt.value} active={value.splits.includes(opt.value)} onClick={() => toggleSplit(opt.value)}>
                  {opt.label}
                </Chip>
              ))}
            </div>
          </div>

          {classificationMeasurements.map((m) => (
            <div key={m.key}>
              <span className={labelClass}>{m.label}</span>
              <div className="flex flex-wrap gap-1.5">
                {classNames(m).map((name) => (
                  <Chip
                    key={name}
                    active={(value.classValues[m.key] ?? []).includes(name)}
                    onClick={() => toggleClassValue(m.key, name)}
                  >
                    {name}
                  </Chip>
                ))}
              </div>
            </div>
          ))}

          {regressionMeasurements.map((m) => (
            <div key={m.key} className="flex items-end gap-2">
              <div className="flex-1">
                <span className={labelClass}>
                  {m.label} min{m.unit ? ` (${m.unit})` : ""}
                </span>
                <AdminInput
                  type="number"
                  placeholder={String(m.min ?? 0)}
                  value={value.ranges[m.key]?.min ?? ""}
                  onChange={(event) => setRange(m.key, "min", event.target.value)}
                />
              </div>
              <div className="flex-1">
                <span className={labelClass}>
                  {m.label} max{m.unit ? ` (${m.unit})` : ""}
                </span>
                <AdminInput
                  type="number"
                  placeholder={String(m.max ?? 100)}
                  value={value.ranges[m.key]?.max ?? ""}
                  onChange={(event) => setRange(m.key, "max", event.target.value)}
                />
              </div>
            </div>
          ))}

          {count > 0 && (
            <div>
              <AdminButton type="button" variant="secondary" onClick={() => onChange(EMPTY_FILTERS)}>
                Clear filters
              </AdminButton>
            </div>
          )}
        </div>
      )}
    </AdminCard>
  );
}
