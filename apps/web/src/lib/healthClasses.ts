// Kept in sync by hand with ml/config.py's CLASS_NAMES — this stack has no
// cross-language (Python <-> TypeScript) shared source of truth.
export const HEALTH_CLASSES = ["Healthy", "Moderate", "Low", "Decay", "Dried", "Disease"] as const;

export type HealthClass = (typeof HEALTH_CLASSES)[number];

export function isHealthClass(value: string): value is HealthClass {
  return (HEALTH_CLASSES as readonly string[]).includes(value);
}
