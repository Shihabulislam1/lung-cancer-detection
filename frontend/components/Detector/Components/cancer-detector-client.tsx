"use client";

import type React from "react";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

// Import our models
import { Report } from "./models/report.model";

// Import our services
import {
  FileService,
  CancerDetectionService,
} from "./services/cancer-detection.service";

// Import our UI components
import { ImageUploader } from "./ui/image-uploader";
import { ProgressIndicator } from "./ui/progress-indicator";
import { UploadButton } from "./ui/upload-button";
import { ReportDisplay } from "./ui/report-display";

/**
 * CancerDetectorClient - Main component orchestrating the cancer detection flow
 * Following SOLID principles:
 * - Single Responsibility: Each component has one job
 * - Open/Closed: Components are extendable without modification
 * - Liskov Substitution: Components can be replaced with subtypes
 * - Interface Segregation: Clean interfaces between components
 * - Dependency Inversion: High-level modules don't depend on low-level modules
 */
export default function CancerDetectorClient() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentReport, setCurrentReport] = useState<Report | null>(null);
  const [activeView, setActiveView] = useState<"upload" | "result">("upload");
  const { toast } = useToast();
  const router = useRouter();

  // Handle file selection
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      // Use the FileService to create a preview
      const previewUrl = await FileService.createPreview(selectedFile);
      setPreview(previewUrl);
    }
  };

  // Handle the upload and analysis process
  const handleUpload = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please select a CT scan image to upload",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    setProgress(0);

    try {
      // Use the CancerDetectionService to upload and process the image
      setIsPredicting(true);
      const report = await CancerDetectionService.uploadImage(
        file,
        setProgress
      );

      setCurrentReport(report);
      setActiveView("result");

      toast({
        title: "Analysis complete",
        description: "Your CT scan has been analyzed",
      });

      router.refresh();
    } catch (error) {
      toast({
        title: "Upload failed",
        description:
          error instanceof Error ? error.message : "Failed to upload image",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
      setIsPredicting(false);
      setProgress(0);
    }
  };

  // Render the appropriate view based on state
  const renderContent = () => {
    if (activeView === "upload" || !currentReport) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center gap-6">
              <ImageUploader
                preview={preview}
                isUploading={isUploading}
                onFileSelect={handleFileChange}
              />

              {file && (
                <div className="text-sm text-gray-500">
                  Selected file: {file.name}
                </div>
              )}

              <ProgressIndicator
                progress={progress}
                isUploading={isUploading}
                isPredicting={isPredicting}
              />

              <UploadButton
                isUploading={isUploading}
                isPredicting={isPredicting}
                hasFile={!!file}
                onUpload={handleUpload}
              />
            </div>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card>
        <CardContent className="pt-6">
          <ReportDisplay report={currentReport} />
          <div className="mt-6 text-center">
            <button
              onClick={() => setActiveView("upload")}
              className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
            >
              Upload another scan
            </button>
          </div>
        </CardContent>
      </Card>
    );
  };

  return renderContent();
}
