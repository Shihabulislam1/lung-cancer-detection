// Report model representing the cancer detection report
export interface Report {
  id: number;
  imageUrl: string;
  status: "pending" | "completed" | "failed";
  rawOutput?: number[][];
  probabilities?: number[][];
  predictedClassIndex?: number;
  createdAt: string;
}

// Analysis result classifications
export const ResultClassification = {
  CANCER_DETECTED: 0,
  NO_CANCER: 1,
  INCONCLUSIVE: 2,
} as const;
