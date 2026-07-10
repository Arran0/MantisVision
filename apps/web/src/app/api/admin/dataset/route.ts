import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { isCondition, isSeverity, isDiseaseSubtype } from "@/lib/taxonomy";
import type { TrainingImage } from "@/lib/types";

const BUCKET = "training-images";
const PAGE_SIZE = 30;
const SIGNED_URL_TTL_S = 60 * 5;

function extensionFor(file: File): string {
  const fromName = file.name.split(".").pop();
  if (fromName && fromName.length <= 5) return fromName.toLowerCase();
  const fromType = file.type.split("/").pop();
  return fromType && fromType.length <= 5 ? fromType.toLowerCase() : "jpg";
}

function numberOrNull(value: FormDataEntryValue | null): number | null {
  if (typeof value !== "string" || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function pctOrNull(value: FormDataEntryValue | null): number | null {
  const parsed = numberOrNull(value);
  if (parsed === null) return null;
  return Math.max(0, Math.min(100, parsed));
}

function stringOrNull(value: FormDataEntryValue | null): string | null {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

export async function POST(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const formData = await request.formData();
  const file = formData.get("file");
  const condition = formData.get("condition");

  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "Missing or invalid 'file' — must be an image." }, { status: 400 });
  }
  if (typeof condition !== "string" || !isCondition(condition)) {
    return NextResponse.json({ error: "'condition' must be one of the known conditions." }, { status: 400 });
  }

  const isBackground = condition === "Background";
  const isDisease = condition === "Disease";

  let severity: string | null = null;
  let subtype: string | null = null;
  let diseaseName: string | null = null;
  if (isDisease) {
    const sev = formData.get("severity");
    const sub = formData.get("subtype");
    if (typeof sev !== "string" || !isSeverity(sev)) {
      return NextResponse.json({ error: "Disease requires a valid 'severity'." }, { status: 400 });
    }
    if (typeof sub !== "string" || !isDiseaseSubtype(sub)) {
      return NextResponse.json({ error: "Disease requires a valid 'subtype'." }, { status: 400 });
    }
    severity = sev;
    subtype = sub;
    // Disease name is free-form but must not contain underscores/spaces that
    // would break the folder-naming convention when the retrain step
    // materializes it into <...>_<Subtype>_<DiseaseName>.
    const rawName = stringOrNull(formData.get("diseaseName"));
    diseaseName = rawName ? rawName.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || null : null;
  }

  const species = isBackground
    ? null
    : stringOrNull(formData.get("species")) ?? "Kappaphycus alvarezii";

  const admin = createAdminClient();
  const storagePath = `${condition}/${crypto.randomUUID()}.${extensionFor(file)}`;

  const { error: uploadError } = await admin.storage
    .from(BUCKET)
    .upload(storagePath, await file.arrayBuffer(), { contentType: file.type });

  if (uploadError) {
    return NextResponse.json({ error: `Upload failed: ${uploadError.message}` }, { status: 502 });
  }

  const gpsLat = numberOrNull(formData.get("gpsLat"));
  const gpsLng = numberOrNull(formData.get("gpsLng"));

  const { data: row, error: insertError } = await admin
    .from("training_images")
    .insert({
      created_by: auth.context.userId,
      storage_path: storagePath,
      condition,
      is_background: isBackground,
      severity,
      subtype,
      disease_name: diseaseName,
      species,
      colour: isBackground ? null : stringOrNull(formData.get("colour")),
      health_score: isBackground ? null : pctOrNull(formData.get("healthScore")),
      dried_pct: isBackground ? null : pctOrNull(formData.get("driedPct")),
      decayed_pct: isBackground ? null : pctOrNull(formData.get("decayedPct")),
      notes: stringOrNull(formData.get("notes")),
      farm: stringOrNull(formData.get("farm")),
      camera: stringOrNull(formData.get("camera")),
      captured_at: stringOrNull(formData.get("capturedAt")),
      water_temperature_c: numberOrNull(formData.get("waterTemperatureC")),
      salinity_ppt: numberOrNull(formData.get("salinityPpt")),
      depth_m: numberOrNull(formData.get("depthM")),
      gps: gpsLat !== null && gpsLng !== null ? `(${gpsLng},${gpsLat})` : null,
    })
    .select()
    .single();

  if (insertError || !row) {
    // Don't leave an orphaned object in storage if the DB insert failed.
    await admin.storage.from(BUCKET).remove([storagePath]);
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
    .select(
      "id, created_at, created_by, species, colour, condition, severity, subtype, disease_name, notes, farm, status, storage_path"
    )
    .order("created_at", { ascending: false })
    .range(from, to);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 502 });
  }

  const images: TrainingImage[] = await Promise.all(
    (rows ?? []).map(async (row) => {
      const { data: signed } = await admin.storage
        .from(BUCKET)
        .createSignedUrl(row.storage_path, SIGNED_URL_TTL_S);
      return {
        id: row.id,
        createdAt: row.created_at,
        createdBy: row.created_by,
        species: row.species,
        colour: row.colour,
        condition: row.condition,
        severity: row.severity,
        subtype: row.subtype,
        diseaseName: row.disease_name,
        notes: row.notes,
        farm: row.farm,
        status: row.status,
        thumbnailUrl: signed?.signedUrl ?? null,
      };
    })
  );

  return NextResponse.json({ images, page });
}
