import { NextRequest, NextResponse } from "next/server";
import { requireContributor } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { getActiveSchema } from "@/lib/serverSchema";
import {
  findMeasurement,
  getPrimaryClassification,
  isValueValidForMeasurement,
  measurementApplies,
  type SchemaDoc,
} from "@/lib/schema";
import type { TrainingImage } from "@/lib/types";

const IMAGES_BUCKET = "training-images";
const MASKS_BUCKET = "training-masks";
const PAGE_SIZE = 10;
const SIGNED_URL_TTL_S = 60 * 5;
const MAX_BULK_FILES = 100;
const UPLOAD_CONCURRENCY = 6;
const DATASET_COLUMNS = "id, created_at, created_by, species, colour, measurements, notes, status, storage_path, split";
const SPLIT_VALUES = new Set(["train", "validation", "test"]);

type DatasetRow = {
  id: string;
  created_at: string;
  created_by: string | null;
  species: string | null;
  colour: string | null;
  measurements: Record<string, string | number> | null;
  notes: string | null;
  status: string;
  storage_path: string;
  split: "train" | "validation" | "test" | null;
};

// A filter clause set built by the Dataset page's filter panel and sent as
// one JSON-encoded `filters` query param (rather than several individually
// named params) since it's an open-ended, schema-driven shape: one entry per
// classification measurement's selected classes, one per regression
// measurement's [min, max], plus an optional split selection.
type DatasetFilterQuery = {
  classValues?: Record<string, string[]>;
  ranges?: Record<string, { min?: number; max?: number }>;
  splits?: string[]; // "train" | "validation" | "test" | "auto"
};

function parseFilters(raw: string | null): DatasetFilterQuery | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return null;
    return parsed as DatasetFilterQuery;
  } catch {
    return null;
  }
}

function rowMatchesFilters(row: DatasetRow, filters: DatasetFilterQuery): boolean {
  const measurements = row.measurements ?? {};
  for (const [key, values] of Object.entries(filters.classValues ?? {})) {
    if (!values || values.length === 0) continue;
    const value = measurements[key];
    if (typeof value !== "string" || !values.includes(value)) return false;
  }
  for (const [key, range] of Object.entries(filters.ranges ?? {})) {
    if (range.min === undefined && range.max === undefined) continue;
    const value = measurements[key];
    if (typeof value !== "number") return false;
    if (range.min !== undefined && value < range.min) return false;
    if (range.max !== undefined && value > range.max) return false;
  }
  if (filters.splits && filters.splits.length > 0) {
    const splitLabel = row.split ?? "auto";
    if (!filters.splits.includes(splitLabel)) return false;
  }
  return true;
}

// Runs `fn` over `items` with at most `limit` calls in flight at once,
// preserving result order. Used for per-file storage uploads, which are
// independent network round-trips and don't need to be serialized.
async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    for (let i = next++; i < items.length; i = next++) {
      const item = items[i] as T;
      results[i] = await fn(item, i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

function extensionFor(file: File): string {
  const fromName = file.name.split(".").pop();
  if (fromName && fromName.length <= 5) return fromName.toLowerCase();
  const fromType = file.type.split("/").pop();
  return fromType && fromType.length <= 5 ? fromType.toLowerCase() : "jpg";
}

function stringOrNull(value: FormDataEntryValue | string | null | undefined): string | null {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

// Legacy flat columns are populated opportunistically from the well-known
// measurement keys (for existing queries/back-compat); they're no longer the
// source of truth for training — `measurements` is. Both the new keys and the
// pre-restructure names are accepted so a mixed dataset keeps filling these
// columns. Species and colour are just schema classifications now
// (measurements["species"] / measurements["colour"]), same as any other — no
// more special-cased form fields or a schema-level "active species".
function measurementString(measurements: Record<string, string | number>, ...keys: string[]): string | null {
  for (const key of keys) if (typeof measurements[key] === "string") return measurements[key] as string;
  return null;
}
function measurementNumber(measurements: Record<string, string | number>, ...keys: string[]): number | null {
  for (const key of keys) if (typeof measurements[key] === "number") return measurements[key] as number;
  return null;
}

// Parses+validates the client-submitted `measurements` JSON field against the
// active schema: each entry must name a real measurement, hold a legal value
// for its type, and satisfy that measurement's applies_when (evaluated
// against the *other* submitted measurement values) — e.g. disease_subtype is
// rejected unless condition was submitted as "Disease". Returns either the
// validated map or a human-readable error.
function validateMeasurements(
  schema: SchemaDoc,
  raw: unknown
): { ok: true; values: Record<string, string | number> } | { ok: false; error: string } {
  if (typeof raw !== "object" || raw === null) return { ok: false, error: "'measurements' must be an object." };
  const values = raw as Record<string, unknown>;

  for (const [key, value] of Object.entries(values)) {
    const measurement = findMeasurement(schema, key);
    if (!measurement) return { ok: false, error: `Unknown measurement ${JSON.stringify(key)}.` };
    if (measurement.type === "segmentation") continue; // segmentation values are filled in from uploaded mask files
    if (!isValueValidForMeasurement(measurement, value))
      return { ok: false, error: `Invalid value for measurement ${JSON.stringify(key)}.` };
    if (!measurementApplies(measurement, values))
      return {
        ok: false,
        error: `Measurement ${JSON.stringify(key)} does not apply given the other submitted values.`,
      };
  }
  return { ok: true, values: values as Record<string, string | number> };
}

export async function POST(request: NextRequest) {
  const auth = await requireContributor();
  if (!auth.ok) return auth.response;

  const formData = await request.formData();
  const files = formData.getAll("files").filter((f): f is File => f instanceof File);
  if (files.length === 0) {
    return NextResponse.json({ error: "Missing 'files' — must include at least one image." }, { status: 400 });
  }
  if (files.length > MAX_BULK_FILES) {
    return NextResponse.json({ error: `Too many files — max ${MAX_BULK_FILES} per upload.` }, { status: 400 });
  }
  const nonImage = files.find((f) => !f.type.startsWith("image/"));
  if (nonImage) {
    return NextResponse.json({ error: `'${nonImage.name}' is not an image.` }, { status: 400 });
  }

  const admin = createAdminClient();
  const schema = await getActiveSchema(admin);

  const measurementsRaw = (() => {
    const raw = formData.get("measurements");
    if (typeof raw !== "string") return {};
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  })();
  if (measurementsRaw === null) {
    return NextResponse.json({ error: "'measurements' must be valid JSON." }, { status: 400 });
  }

  const validated = validateMeasurements(schema, measurementsRaw);
  if (!validated.ok) {
    return NextResponse.json({ error: validated.error }, { status: 400 });
  }
  const measurements: Record<string, string | number> = { ...validated.values };

  // Upload any segmentation mask files (submitted as `mask:<measurementKey>`)
  // to the training-masks bucket, then store the resulting path as that
  // measurement's value. A mask is specific to a single image's pixels, so
  // it's only accepted (and only sent by the client) when uploading exactly
  // one photo.
  if (files.length === 1) {
    for (const measurement of schema.measurements) {
      if (measurement.type !== "segmentation") continue;
      if (!measurementApplies(measurement, measurements)) continue;
      const maskFile = formData.get(`mask:${measurement.key}`);
      if (!(maskFile instanceof File)) continue;
      const maskPath = `${measurement.key}/${crypto.randomUUID()}.${extensionFor(maskFile)}`;
      const { error: maskUploadError } = await admin.storage
        .from(MASKS_BUCKET)
        .upload(maskPath, await maskFile.arrayBuffer(), { contentType: maskFile.type });
      if (maskUploadError) {
        return NextResponse.json({ error: `Mask upload failed: ${maskUploadError.message}` }, { status: 502 });
      }
      measurements[measurement.key] = maskPath;
    }
  }

  const primary = getPrimaryClassification(schema);
  const primaryValue = primary ? (measurements[primary.key] as string | undefined) : undefined;
  const isBackground = !!primary && primaryValue === primary.background_class;
  const notes = stringOrNull(formData.get("notes"));

  // The labels (measurements/notes) are shared across the whole batch — each
  // file just gets its own storage object and its own `training_images` row.
  // Uploads run with bounded concurrency (they're independent network calls),
  // then all rows are written in a single insert instead of one per file.
  // One file's storage upload failing doesn't stop the rest.
  type UploadOutcome =
    | { ok: true; index: number; file: File; storagePath: string }
    | { ok: false; index: number; file: File; error: string };

  const outcomes = await mapWithConcurrency<File, UploadOutcome>(files, UPLOAD_CONCURRENCY, async (file, index) => {
    const storagePath = `${primaryValue ?? "uncategorized"}/${crypto.randomUUID()}.${extensionFor(file)}`;
    const { error: uploadError } = await admin.storage
      .from(IMAGES_BUCKET)
      .upload(storagePath, await file.arrayBuffer(), { contentType: file.type });
    if (uploadError) {
      return { ok: false, index, file, error: `Upload failed: ${uploadError.message}` };
    }
    return { ok: true, index, file, storagePath };
  });

  const results: { file: string; id?: string; error?: string }[] = new Array(files.length);
  for (const outcome of outcomes) {
    if (!outcome.ok) results[outcome.index] = { file: outcome.file.name, error: outcome.error };
  }
  const uploaded = outcomes.filter((o): o is Extract<UploadOutcome, { ok: true }> => o.ok);

  if (uploaded.length > 0) {
    const { data: rows, error: insertError } = await admin
      .from("training_images")
      .insert(
        uploaded.map(({ storagePath }) => ({
          created_by: auth.context.userId,
          storage_path: storagePath,
          measurements,
          condition: measurementString(measurements, "health_status", "condition"),
          subtype: measurementString(measurements, "disease", "disease_subtype"),
          health_score: measurementNumber(measurements, "health_score"),
          dried_pct: measurementNumber(measurements, "dried", "dried_extent"),
          decayed_pct: measurementNumber(measurements, "decayed", "decayed_extent"),
          is_background: isBackground,
          species: measurementString(measurements, "species"),
          colour: measurementString(measurements, "colour"),
          notes,
        }))
      )
      .select();

    if (insertError || !rows || rows.length !== uploaded.length) {
      // Don't leave orphaned objects in storage if the batch insert failed.
      await Promise.all(uploaded.map(({ storagePath }) => admin.storage.from(IMAGES_BUCKET).remove([storagePath])));
      for (const { index, file } of uploaded) {
        results[index] = { file: file.name, error: `Failed to save the label: ${insertError?.message ?? "unknown error"}` };
      }
    } else {
      uploaded.forEach(({ index, file }, i) => {
        results[index] = { file: file.name, id: (rows[i] as (typeof rows)[number]).id };
      });
    }
  }

  const allFailed = results.every((r) => r.error);
  return NextResponse.json({ results }, { status: allFailed ? 502 : 201 });
}

export async function GET(request: NextRequest) {
  const auth = await requireContributor();
  if (!auth.ok) return auth.response;

  const page = Math.max(1, Number(request.nextUrl.searchParams.get("page") ?? "1") || 1);
  const filters = parseFilters(request.nextUrl.searchParams.get("filters"));

  const admin = createAdminClient();

  let rows: DatasetRow[];
  let total: number;

  if (filters) {
    // Filtering on values nested inside the `measurements` jsonb column
    // (and, for a range, needing a numeric comparison) doesn't push down
    // cleanly through a single PostgREST query for an open-ended,
    // schema-driven filter set, so fetch the (still modest-sized) full
    // dataset and filter/paginate in memory instead.
    const { data, error } = await admin
      .from("training_images")
      .select(DATASET_COLUMNS)
      .order("created_at", { ascending: false });
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 502 });
    }
    const matched = ((data as DatasetRow[] | null) ?? []).filter((row) => rowMatchesFilters(row, filters));
    total = matched.length;
    rows = matched.slice((page - 1) * PAGE_SIZE, (page - 1) * PAGE_SIZE + PAGE_SIZE);
  } else {
    const from = (page - 1) * PAGE_SIZE;
    const to = from + PAGE_SIZE - 1;
    const {
      data,
      error,
      count,
    } = await admin.from("training_images").select(DATASET_COLUMNS, { count: "exact" }).order("created_at", { ascending: false }).range(from, to);
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 502 });
    }
    rows = (data as DatasetRow[] | null) ?? [];
    total = count ?? 0;
  }

  const images: TrainingImage[] = await Promise.all(
    rows.map(async (row) => {
      const { data: signed } = await admin.storage
        .from(IMAGES_BUCKET)
        .createSignedUrl(row.storage_path, SIGNED_URL_TTL_S);
      return {
        id: row.id,
        createdAt: row.created_at,
        createdBy: row.created_by,
        species: row.species,
        colour: row.colour,
        measurements: (row.measurements as Record<string, string | number>) ?? {},
        notes: row.notes,
        status: row.status,
        split: row.split ?? null,
        thumbnailUrl: signed?.signedUrl ?? null,
      };
    })
  );

  const hasMore = page * PAGE_SIZE < total;

  return NextResponse.json({ images, page, pageSize: PAGE_SIZE, total, hasMore });
}

// Edits the feature-column (measurement) values and notes of an
// already-uploaded photo — e.g. fixing a mislabeled class after the fact —
// without re-uploading the image itself.
export async function PATCH(request: NextRequest) {
  const auth = await requireContributor();
  if (!auth.ok) return auth.response;

  const body = await request.json().catch(() => null);
  const id = typeof body?.id === "string" ? body.id : "";
  if (!id) {
    return NextResponse.json({ error: "Missing 'id'." }, { status: 400 });
  }

  const admin = createAdminClient();
  const schema = await getActiveSchema(admin);

  const validated = validateMeasurements(schema, body?.measurements ?? {});
  if (!validated.ok) {
    return NextResponse.json({ error: validated.error }, { status: 400 });
  }
  const measurements = validated.values;

  // `split` pins this image to a specific retrain split (train/validation/
  // test); anything else — omitted, null, or the sentinel "auto" — clears
  // the pin back to "assign automatically" (ml/scripts/split_dataset.py's
  // random ratio-based split).
  let split: "train" | "validation" | "test" | null = null;
  if (body?.split !== undefined && body?.split !== null && body?.split !== "auto") {
    if (typeof body.split !== "string" || !SPLIT_VALUES.has(body.split)) {
      return NextResponse.json({ error: "'split' must be one of train, validation, test, or auto." }, { status: 400 });
    }
    split = body.split as "train" | "validation" | "test";
  }

  const primary = getPrimaryClassification(schema);
  const primaryValue = primary ? (measurements[primary.key] as string | undefined) : undefined;
  const isBackground = !!primary && primaryValue === primary.background_class;

  const { data: row, error: updateError } = await admin
    .from("training_images")
    .update({
      measurements,
      condition: measurementString(measurements, "health_status", "condition"),
      subtype: measurementString(measurements, "disease", "disease_subtype"),
      health_score: measurementNumber(measurements, "health_score"),
      dried_pct: measurementNumber(measurements, "dried", "dried_extent"),
      decayed_pct: measurementNumber(measurements, "decayed", "decayed_extent"),
      is_background: isBackground,
      species: measurementString(measurements, "species"),
      colour: measurementString(measurements, "colour"),
      notes: stringOrNull(body?.notes),
      split,
    })
    .eq("id", id)
    .select("id, species, colour, measurements, notes, split")
    .single();

  if (updateError || !row) {
    return NextResponse.json(
      { error: `Failed to save the edit: ${updateError?.message ?? "photo not found"}` },
      { status: 502 }
    );
  }

  return NextResponse.json({
    id: row.id,
    species: row.species,
    colour: row.colour,
    measurements: (row.measurements as Record<string, string | number>) ?? {},
    notes: row.notes,
    split: row.split ?? null,
  });
}
