export interface MeasurementResult {
  type: "classification" | "regression" | "segmentation";
  label: string;
  value: string | number | null;
  confidence: number | null; // classification only
  explanation: string | null;
  recommendation: string | null;
  unit: string | null; // regression only
  min: number | null; // regression only
  max: number | null; // regression only
  coverage: Record<string, number> | null; // segmentation only: {seg_class_name: pct_of_frame}
  segColors: Record<string, string> | null; // segmentation only: {seg_class_name: hex color}
  maskPngBase64: string | null; // segmentation only
  gradcamPngBase64: string | null; // classification only
}

export interface PredictionResult {
  // Every measurement the active schema defines, keyed by measurement key
  // and in schema order. No flat/legacy fields — render whichever entries
  // have a non-null value.
  measurements: Record<string, MeasurementResult>;
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
