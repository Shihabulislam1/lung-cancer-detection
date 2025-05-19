"use client";

import React from "react";
import { Progress } from "@/components/ui/progress";

interface ProgressIndicatorProps {
  progress: number;
  isUploading: boolean;
  isPredicting: boolean;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  isUploading,
  isPredicting,
}) => {
  const getMessage = () => {
    if (!isUploading && !isPredicting) return null;

    if (progress < 50) return "Uploading image...";
    if (progress < 100) return "Processing and analyzing...";
    return "Finalizing results...";
  };

  if (!isUploading && !isPredicting) return null;

  return (
    <div className="w-full max-w-md">
      <Progress value={progress} className="h-2 mb-2" />
      <p className="text-sm text-gray-500 text-center">{getMessage()}</p>
    </div>
  );
};
