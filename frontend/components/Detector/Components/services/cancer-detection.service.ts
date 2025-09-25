// Services for handling file uploading and cancer detection API
import type React from "react";
import { Report } from "../models/report.model";

export class FileService {
  static createPreview(file: File): Promise<string> {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        resolve(reader.result as string);
      };
      reader.readAsDataURL(file);
    });
  }
}

export class CancerDetectionService {
  static async uploadImage(
    file: File,
    onProgressUpdate: React.Dispatch<React.SetStateAction<number>>
  ): Promise<Report> {
    // Start progress simulation
    const progressInterval = setInterval(() => {
      onProgressUpdate((prev: number) => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return 95;
        }
        return prev + 5;
      });
    }, 500);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Add timeout to fetch to prevent hanging indefinitely
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      // Call FastAPI endpoint directly
      const mlEndpoint =
        process.env.NEXT_PUBLIC_ML_API_URL?.replace(/\/$/, "") ||
        "http://localhost:8000/predict";

      let response: Response;
      try {
        response = await fetch(mlEndpoint, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      } catch (netErr) {
        clearTimeout(timeoutId);
        throw new Error(
          netErr instanceof Error
            ? `Network error: ${netErr.message}`
            : "Network error: unable to reach prediction service"
        );
      }

      clearTimeout(timeoutId);
      clearInterval(progressInterval);
      onProgressUpdate(100);

      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "Unknown error" }));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      // Parse FastAPI response format
      const mlJson = await response.json();
      if (mlJson && mlJson.ok === false) {
        throw new Error(mlJson.error || "Prediction failed");
      }

      const prediction = mlJson?.prediction;
      const probabilitiesObj: Record<string, number> =
        prediction?.probabilities || {};

      // Normalize keys (handle different casing / naming like Bengin vs Benign)
      const norm = (k: string) =>
        k.toLowerCase().replace(/benign|bengin/, "benign");
      let malignant = 0,
        normal = 0,
        benign = 0;
      for (const [k, v] of Object.entries(probabilitiesObj)) {
        const nk = norm(k);
        if (nk.includes("malig")) malignant = v;
        else if (nk.includes("normal")) normal = v;
        else if (nk.includes("benign")) benign = v;
      }

      const ordered = [malignant, normal, benign];
      const predictedClassIndex = ordered.indexOf(Math.max(...ordered));

      // Fallback if all zero (e.g., unexpected response)
      if (ordered.every((v) => v === 0)) {
        console.warn(
          "All probabilities zero or unmapped. Raw response:",
          mlJson
        );
      }

      // Create a preview URL for the uploaded file
      const imageUrl = URL.createObjectURL(file);

      // Persist the report via internal API (server will attach real id)
      let persisted: Report | null = null;
      try {
        const persistRes = await fetch("/api/cancer-detection/reports", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            imageUrl, // For now use the local object URL; later could replace with uploaded URL
            probabilities: [ordered],
            predictedClassIndex,
            rawOutput: prediction?.logits ? [prediction.logits] : undefined,
          }),
        });
        if (persistRes.ok) {
          persisted = (await persistRes.json()) as Report;
        } else {
          console.warn("Failed to persist report", await persistRes.text());
        }
      } catch (e) {
        console.warn(
          "Persistence error (report still available in session)",
          e
        );
      }

      return (persisted || {
        id: -1,
        imageUrl,
        status: "completed",
        probabilities: [ordered],
        predictedClassIndex,
        createdAt: new Date().toISOString(),
      }) as Report;
    } catch (error) {
      clearInterval(progressInterval);
      console.error("Upload error:", error);

      if (error instanceof Error && error.name === "AbortError") {
        throw new Error("Request timed out. Please try again.");
      }

      throw error instanceof Error
        ? error
        : new Error("Failed to upload image. Please try again later.");
    }
  }

  static getResultMessage(index?: number): string {
    if (index === undefined) return "Results unavailable";

    switch (index) {
      case 0:
        return "Malignant - Cancer detected";
      case 1:
        return "Normal - No cancer detected";
      case 2:
        return "Benign - Non-cancerous growth";
      default:
        return "Unknown result";
    }
  }

  static getResultColor(index?: number): string {
    if (index === undefined) return "text-gray-500";

    switch (index) {
      case 0:
        return "text-red-600";
      case 1:
        return "text-green-600";
      case 2:
        return "text-blue-600";
      default:
        return "text-gray-500";
    }
  }
}
