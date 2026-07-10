import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import { createAdminClient } from "@/lib/supabase/admin";
import { isHealthClass } from "@/lib/healthClasses";
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

function stringOrNull(value: FormDataEntryValue | null): string | null {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

export async function POST(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const formData = await request.formData();
  const file = formData.get("file");
  const health = formData.get("health");

  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "Missing or invalid 'file' — must be an image." }, { status: 400 });
  }
  if (typeof health !== "string" || !isHealthClass(health)) {
    return NextResponse.json({ error: "'health' must be one of the known health classes." }, { status: 400 });
  }

  const species = stringOrNull(formData.get("species")) ?? "Kappaphycus alvarezii";
  const colour = stringOrNull(formData.get("colour"));
  const notes = stringOrNull(formData.get("notes"));
  const farm = stringOrNull(formData.get("farm"));
  const camera = stringOrNull(formData.get("camera"));
  const capturedAt = stringOrNull(formData.get("capturedAt"));
  const waterTemperatureC = numberOrNull(formData.get("waterTemperatureC"));
  const salinityPpt = numberOrNull(formData.get("salinityPpt"));
  const depthM = numberOrNull(formData.get("depthM"));
  const gpsLat = numberOrNull(formData.get("gpsLat"));
  const gpsLng = numberOrNull(formData.get("gpsLng"));

  const admin = createAdminClient();
  const storagePath = `${health}/${crypto.randomUUID()}.${extensionFor(file)}`;

  const { error: uploadError } = await admin.storage
    .from(BUCKET)
    .upload(storagePath, await file.arrayBuffer(), { contentType: file.type });

  if (uploadError) {
    return NextResponse.json({ error: `Upload failed: ${uploadError.message}` }, { status: 502 });
  }

  const { data: row, error: insertError } = await admin
    .from("training_images")
    .insert({
      created_by: auth.context.userId,
      storage_path: storagePath,
      species,
      colour,
      health,
      notes,
      farm,
      camera,
      captured_at: capturedAt,
      water_temperature_c: waterTemperatureC,
      salinity_ppt: salinityPpt,
      depth_m: depthM,
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
    .select("id, created_at, created_by, species, colour, health, notes, farm, status, storage_path")
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
        health: row.health,
        notes: row.notes,
        farm: row.farm,
        status: row.status,
        thumbnailUrl: signed?.signedUrl ?? null,
      };
    })
  );

  return NextResponse.json({ images, page });
}
