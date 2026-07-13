// Kept in sync by hand with ml/config.py (CONDITION_CLASSES, DISEASE_SUBTYPES,
// SEVERITIES) and the folder-naming convention in ml/src/data/labels.py —
// this stack has no cross-language (Python <-> TypeScript) shared source.

// Conditions the admin can label. Background is the negative "no seaweed" class.
export const CONDITIONS = ["Healthy", "Disease", "Decay", "Dried", "Background"] as const;
export type Condition = (typeof CONDITIONS)[number];

export const SEVERITIES = ["Moderate", "Low"] as const;
export type Severity = (typeof SEVERITIES)[number];

export const DISEASE_SUBTYPES = ["IceIce", "Epiphyte", "Bacterial", "Bleaching", "Unknown"] as const;
export type DiseaseSubtype = (typeof DISEASE_SUBTYPES)[number];

export function isCondition(value: string): value is Condition {
  return (CONDITIONS as readonly string[]).includes(value);
}
export function isSeverity(value: string): value is Severity {
  return (SEVERITIES as readonly string[]).includes(value);
}
export function isDiseaseSubtype(value: string): value is DiseaseSubtype {
  return (DISEASE_SUBTYPES as readonly string[]).includes(value);
}

// Kept in sync by hand with ml/config.py's SPECIES["slug"].
export const SPECIES_SLUG = "Kappaphycus_alvarezii";

// Decay/Dried only ever use "Low" severity — see ml/config.py's
// FIXED_SEVERITY_CONDITIONS and ml/src/data/labels.py.
const FIXED_SEVERITY: Partial<Record<Condition, Severity>> = { Decay: "Low", Dried: "Low" };

// TypeScript mirror of ml/src/data/labels.py's build_class_folder — the admin
// upload route uses this to name the staged file's folder on GitHub so the
// retrain job can materialize it with zero translation.
export function buildClassFolder(
  condition: Condition,
  severity?: Severity | null,
  subtype?: DiseaseSubtype | null,
  diseaseName?: string | null
): string {
  if (condition === "Background") return "Background";
  if (condition === "Healthy") return `${SPECIES_SLUG}_Healthy`;

  const fixedSeverity = FIXED_SEVERITY[condition];
  if (fixedSeverity) return `${SPECIES_SLUG}_${fixedSeverity}_${condition}`;

  if (condition === "Disease") {
    if (!severity || !subtype) {
      throw new Error("Disease requires both severity and subtype.");
    }
    const tokens = [SPECIES_SLUG, severity, "Disease", subtype];
    if (diseaseName) tokens.push(diseaseName);
    return tokens.join("_");
  }

  throw new Error(`Unknown condition ${condition}.`);
}
