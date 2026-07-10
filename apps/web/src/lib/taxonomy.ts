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
