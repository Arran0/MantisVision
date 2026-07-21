export interface PredictionResult {
  species: string;
  isSeaweed: boolean;
  condition: string;
  // Derived display level (Healthy/Moderate/Low). null for Background/no seaweed.
  health: string | null;
  healthScore: number | null; // 0-100
  confidence: number;
  diseaseSubtype: string | null;
  driedPct: number | null;
  decayedPct: number | null;
  explanation: string;
  recommendation: string;
  gradcamPngBase64: string;
}

export interface TrainingImage {
  id: string;
  createdAt: string;
  createdBy: string | null;
  species: string | null;
  colour: string | null;
  // Per-image measurement values keyed by measurement key (class name string,
  // numeric value, or — for a segmentation measurement — a mask storage path).
  measurements: Record<string, string | number>;
  notes: string | null;
  status: string;
  // Admin-chosen train/validation/test pin for the retrain split. null means
  // "assign automatically" (the retrain job's random ratio-based split).
  split: "train" | "validation" | "test" | null;
  thumbnailUrl: string | null;
}

export interface TeamMember {
  id: string;
  email: string | null;
  role: "admin" | "contributor" | "viewer";
  createdAt: string;
  isSelf: boolean;
}

export interface ModelRun {
  id: string;
  createdAt: string;
  status: "queued" | "running" | "completed" | "failed";
  githubRunId: string | null;
  datasetImageCount: number | null;
  metrics: Record<string, unknown> | null;
  checkpointUrl: string | null;
  error: string | null;
  promotedAt: string | null;
}
