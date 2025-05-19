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

      const response = await fetch("/api/cancer-detection", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      clearInterval(progressInterval);
      onProgressUpdate(100);

      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "Unknown error" }));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      return await response.json();
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
        return "High probability of cancer detected";
      case 1:
        return "No cancer detected";
      case 2:
        return "Inconclusive results - further testing recommended";
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
        return "text-amber-500";
      default:
        return "text-gray-500";
    }
  }
}
