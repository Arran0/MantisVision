export interface PredictionResult {
  species: string;
  health: string;
  confidence: number;
  explanation: string;
  recommendation: string;
  gradcamPngBase64: string;
}

export interface TrainingImage {
  id: string;
  createdAt: string;
  createdBy: string | null;
  species: string;
  colour: string | null;
  health: string;
  notes: string | null;
  farm: string | null;
  status: string;
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
