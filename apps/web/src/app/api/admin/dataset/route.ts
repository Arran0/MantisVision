import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/supabase/require-admin";
import {
  isCondition,
  isSeverity,
  isDiseaseSubtype,
  buildClassFolder,
  type Severity,
  type DiseaseSubtype,
} from "@/lib/taxonomy";
import { ensureStagingBranch, getFile, putFile, listDir } from "@/lib/github";
import type { TrainingImage } from "@/lib/types";

// Staged uploads land here on the dataset-staging branch (never on main) as
// <STAGING_ROOT>/<class_folder>/<uuid>.<ext>, with one metadata.csv row per
// image. The retrain job checks this branch out, moves the images into the
// real dataset/<slug>/<split>/<class_folder>/ tree, archives them to Kaggle,
// then resets this branch back to main's tip — see
// ml/scripts/retrain_and_report.py.
const STAGING_ROOT = "ml/dataset_incoming";
const METADATA_PATH = `${STAGING_ROOT}/metadata.csv`;

const CSV_COLUMNS = [
  "image_path",
  "condition",
  "severity",
  "subtype",
  "disease_name",
  "species",
  "colour",
  "notes",
  "farm",
  "camera",
  "captured_at",
  "water_temperature_c",
  "salinity_ppt",
  "depth_m",
  "gps_lat",
  "gps_lng",
  "health_score",
  "dried_pct",
  "decayed_pct",
  "uploaded_by",
  "uploaded_at",
] as const;

function csvEscape(value: string): string {
  if (value === "") return "";
  if (/[",\n]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

function toCsvRow(fields: Record<(typeof CSV_COLUMNS)[number], string>): string {
  return CSV_COLUMNS.map((col) => csvEscape(fields[col])).join(",") + "\n";
}

// Minimal CSV parser — good enough for our own controlled output (quoted
// fields with escaped "" only), not a general-purpose CSV library.
function parseCsv(content: string): Record<string, string>[] {
  const lines = content.split(/\r?\n/).filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const header = splitCsvLine(lines[0]!);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    const row: Record<string, string> = {};
    header.forEach((col, i) => (row[col] = values[i] ?? ""));
    return row;
  });
}

function splitCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (inQuotes) {
      if (char === '"' && line[i + 1] === '"') {
        current += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        current += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function extensionFor(file: File): string {
  const fromName = file.name.split(".").pop();
  if (fromName && fromName.length <= 5) return fromName.toLowerCase();
  const fromType = file.type.split("/").pop();
  return fromType && fromType.length <= 5 ? fromType.toLowerCase() : "jpg";
}

function stringOrEmpty(value: FormDataEntryValue | null): string {
  return typeof value === "string" ? value.trim() : "";
}

function pctOrEmpty(value: FormDataEntryValue | null): string {
  if (typeof value !== "string" || value.trim() === "") return "";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "";
  return String(Math.max(0, Math.min(100, parsed)));
}

export async function POST(request: NextRequest) {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  const formData = await request.formData();
  const file = formData.get("file");
  const conditionRaw = formData.get("condition");

  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "Missing or invalid 'file' — must be an image." }, { status: 400 });
  }
  if (typeof conditionRaw !== "string" || !isCondition(conditionRaw)) {
    return NextResponse.json({ error: "'condition' must be one of the known conditions." }, { status: 400 });
  }
  const condition = conditionRaw;
  const isBackground = condition === "Background";
  const isDisease = condition === "Disease";

  let severity: Severity | null = null;
  let subtype: DiseaseSubtype | null = null;
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
    const rawName = stringOrEmpty(formData.get("diseaseName"));
    diseaseName = rawName ? rawName.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || null : null;
  }

  let classFolder: string;
  try {
    classFolder = buildClassFolder(condition, severity, subtype, diseaseName);
  } catch (err) {
    return NextResponse.json({ error: err instanceof Error ? err.message : "Invalid label." }, { status: 400 });
  }

  const filename = `${crypto.randomUUID()}.${extensionFor(file)}`;
  const imagePath = `${classFolder}/${filename}`;

  try {
    await ensureStagingBranch();
    await putFile(
      `${STAGING_ROOT}/${imagePath}`,
      Buffer.from(await file.arrayBuffer()),
      `Label: ${imagePath}`
    );

    const row = toCsvRow({
      image_path: imagePath,
      condition,
      severity: severity ?? "",
      subtype: subtype ?? "",
      disease_name: diseaseName ?? "",
      species: isBackground ? "" : stringOrEmpty(formData.get("species")) || "Kappaphycus alvarezii",
      colour: isBackground ? "" : stringOrEmpty(formData.get("colour")),
      notes: stringOrEmpty(formData.get("notes")),
      farm: stringOrEmpty(formData.get("farm")),
      camera: stringOrEmpty(formData.get("camera")),
      captured_at: stringOrEmpty(formData.get("capturedAt")),
      water_temperature_c: stringOrEmpty(formData.get("waterTemperatureC")),
      salinity_ppt: stringOrEmpty(formData.get("salinityPpt")),
      depth_m: stringOrEmpty(formData.get("depthM")),
      gps_lat: stringOrEmpty(formData.get("gpsLat")),
      gps_lng: stringOrEmpty(formData.get("gpsLng")),
      health_score: isBackground ? "" : pctOrEmpty(formData.get("healthScore")),
      dried_pct: isBackground ? "" : pctOrEmpty(formData.get("driedPct")),
      decayed_pct: isBackground ? "" : pctOrEmpty(formData.get("decayedPct")),
      uploaded_by: auth.context.email ?? "",
      uploaded_at: new Date().toISOString(),
    });

    // One retry on a concurrent-write conflict (stale sha) — good enough for
    // a small admin team; a genuinely busy uploader can just resubmit.
    let lastError: unknown;
    for (let attempt = 0; attempt < 2; attempt++) {
      const existing = await getFile(METADATA_PATH);
      const header = CSV_COLUMNS.join(",") + "\n";
      const newContent = existing ? existing.content + row : header + row;
      try {
        await putFile(METADATA_PATH, newContent, `Add metadata for ${imagePath}`, undefined, existing?.sha);
        lastError = null;
        break;
      } catch (err) {
        lastError = err;
      }
    }
    if (lastError) throw lastError;
  } catch (err) {
    return NextResponse.json(
      { error: `Failed to save the label: ${err instanceof Error ? err.message : "unknown error"}` },
      { status: 502 }
    );
  }

  return NextResponse.json({ path: imagePath }, { status: 201 });
}

export async function GET() {
  const auth = await requireAdmin();
  if (!auth.ok) return auth.response;

  try {
    const metadata = await getFile(METADATA_PATH);
    const rows = metadata ? parseCsv(metadata.content) : [];

    // metadata.csv is append-only from POST; confirm the file still exists in
    // the staging tree (it's removed once the retrain job clears the branch).
    const staged = await listDir(STAGING_ROOT);
    const stillStaged = new Set(staged.filter((e) => e.type === "dir").map((e) => e.name));

    const images: TrainingImage[] = rows
      .filter((row) => stillStaged.has((row.image_path ?? "").split("/")[0] ?? ""))
      .reverse() // newest first
      .map((row) => ({
        id: row.image_path ?? "",
        createdAt: row.uploaded_at ?? "",
        createdBy: row.uploaded_by || null,
        species: row.species || null,
        colour: row.colour || null,
        condition: row.condition ?? "",
        severity: row.severity || null,
        subtype: row.subtype || null,
        diseaseName: row.disease_name || null,
        notes: row.notes || null,
        farm: row.farm || null,
        thumbnailUrl: `/api/admin/dataset/thumbnail?path=${encodeURIComponent(row.image_path ?? "")}`,
      }));

    return NextResponse.json({ images });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to list the dataset." },
      { status: 502 }
    );
  }
}
