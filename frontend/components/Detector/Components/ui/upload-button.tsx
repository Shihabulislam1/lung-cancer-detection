"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Loader2, Upload } from "lucide-react";

interface UploadButtonProps {
  isUploading: boolean;
  isPredicting: boolean;
  hasFile: boolean;
  onUpload: () => void;
}

export const UploadButton: React.FC<UploadButtonProps> = ({
  isUploading,
  isPredicting,
  hasFile,
  onUpload,
}) => {
  const isProcessing = isUploading || isPredicting;

  return (
    <Button
      onClick={onUpload}
      disabled={!hasFile || isProcessing}
      className="w-full max-w-md"
    >
      {isProcessing ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Processing...
        </>
      ) : (
        <>
          <Upload className="mr-2 h-4 w-4" />
          Upload for Analysis
        </>
      )}
    </Button>
  );
};
