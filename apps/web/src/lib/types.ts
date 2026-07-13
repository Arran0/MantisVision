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
  // Relative path under the dataset-staging branch's ml/dataset_incoming/,
  // e.g. "Kappaphycus_alvarezii_Healthy/<uuid>.jpg" — doubles as a unique id.
  id: string;
  createdAt: string;
  createdBy: string | null;
  species: string | null;
  colour: string | null;
  condition: string;
  severity: string | null;
  subtype: string | null;
  diseaseName: string | null;
  notes: string | null;
  farm: string | null;
  thumbnailUrl: string | null;
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
