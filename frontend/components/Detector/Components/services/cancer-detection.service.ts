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

      // Always call our internal Next.js API route.
      // That route handles auth, persistence, and securely calls the ML backend
      // (Hugging Face) with its Bearer token server-side.
      const apiEndpoint = "/api/cancer-detection";

      let response: Response;
      try {
        response = await fetch(apiEndpoint, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
      } catch (netErr) {
        clearTimeout(timeoutId);
        throw new Error(
          netErr instanceof Error
            ? `Network error: ${netErr.message}`
            : "Network error: unable to reach server"
        );
      }

      clearTimeout(timeoutId);
      clearInterval(progressInterval);
      onProgressUpdate(100);

      if (!response.ok) {
        // Next route returns { error: string } on failures.
        const errorData = await response
          .json()
          .catch(() => ({ error: "Unknown error" }));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      // Response is the persisted report from our server.
      return (await response.json()) as Report;
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
