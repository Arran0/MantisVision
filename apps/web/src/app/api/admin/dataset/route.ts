import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
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
const PAGE_SIZE = 30;
const SIGNED_URL_TTL_S = 60 * 5;

function extensionFor(file: File): string {
  const fromName = file.name.split(".").pop();
  if (fromName && fromName.length <= 5) return fromName.toLowerCase();
  const fromType = file.type.split("/").pop();
  return fromType && fromType.length <= 5 ? fromType.toLowerCase() : "jpg";
}

function stringOrNull(value: FormDataEntryValue | null): string | null {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
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
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const formData = await request.formData();
  const file = formData.get("file");
  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "Missing or invalid 'file' — must be an image." }, { status: 400 });
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
  // measurement's value.
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

  const primary = getPrimaryClassification(schema);
  const primaryValue = primary ? (measurements[primary.key] as string | undefined) : undefined;
  const isBackground = !!primary && primaryValue === primary.background_class;

  const activeSpeciesName =
    schema.species.find((s) => s.slug === schema.active_species_slug)?.name ?? schema.species[0]?.name ?? null;
  const species = isBackground ? null : stringOrNull(formData.get("species")) ?? activeSpeciesName;

  const storagePath = `${primaryValue ?? "uncategorized"}/${crypto.randomUUID()}.${extensionFor(file)}`;

  const { error: uploadError } = await admin.storage
    .from(IMAGES_BUCKET)
    .upload(storagePath, await file.arrayBuffer(), { contentType: file.type });

  if (uploadError) {
    return NextResponse.json({ error: `Upload failed: ${uploadError.message}` }, { status: 502 });
  }

  // Legacy flat columns are populated opportunistically from the well-known
  // measurement keys (for existing queries/back-compat); they're no longer
  // the source of truth for training — `measurements` is. Both the new keys
  // and the pre-restructure names are accepted so a mixed dataset keeps
  // filling these columns.
  const measurementString = (...keys: string[]): string | null => {
    for (const key of keys) if (typeof measurements[key] === "string") return measurements[key] as string;
    return null;
  };
  const measurementNumber = (...keys: string[]): number | null => {
    for (const key of keys) if (typeof measurements[key] === "number") return measurements[key] as number;
    return null;
  };
  const { data: row, error: insertError } = await admin
    .from("training_images")
    .insert({
      created_by: auth.context.userId,
      storage_path: storagePath,
      measurements,
      condition: measurementString("health_status", "condition"),
      subtype: measurementString("disease", "disease_subtype"),
      health_score: measurementNumber("health_score"),
      dried_pct: measurementNumber("dried", "dried_extent"),
      decayed_pct: measurementNumber("decayed", "decayed_extent"),
      is_background: isBackground,
      species,
      // Colour is now a schema classification (measurements["colour"]); fall
      // back to a legacy free-text colour field if one is still submitted.
      colour: isBackground ? null : measurementString("colour") ?? stringOrNull(formData.get("colour")),
      notes: stringOrNull(formData.get("notes")),
    })
    .select()
    .single();

  if (insertError || !row) {
    // Don't leave an orphaned object in storage if the DB insert failed.
    await admin.storage.from(IMAGES_BUCKET).remove([storagePath]);
    return NextResponse.json(
      { error: `Failed to save the label: ${insertError?.message ?? "unknown error"}` },
      { status: 502 }
    );
  }

  return NextResponse.json({ id: row.id }, { status: 201 });
}

export async function GET(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const page = Math.max(1, Number(request.nextUrl.searchParams.get("page") ?? "1") || 1);
  const from = (page - 1) * PAGE_SIZE;
  const to = from + PAGE_SIZE - 1;

  const admin = createAdminClient();
  const { data: rows, error } = await admin
    .from("training_images")
    .select("id, created_at, created_by, species, colour, measurements, notes, status, storage_path")
    .order("created_at", { ascending: false })
    .range(from, to);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  const images: TrainingImage[] = await Promise.all(
    (rows ?? []).map(async (row) => {
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
        thumbnailUrl: signed?.signedUrl ?? null,
      };
    })
  );

  return NextResponse.json({ images, page });
}
