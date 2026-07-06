export interface PredictionResult {
  species: string;
  health: string;
  confidence: number;
  explanation: string;
  recommendation: string;
  gradcamPngBase64: string;
}
