"use client";

import React from "react";
import Image from "next/image";
import { FileImage } from "lucide-react";

interface ImageUploaderProps {
  preview: string | null;
  isUploading: boolean;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  preview,
  isUploading,
  onFileSelect,
}) => {
  return (
    <div
      className="border-2 border-dashed border-gray-300 rounded-lg p-12 w-full max-w-md flex flex-col items-center justify-center cursor-pointer hover:border-gray-400 transition-colors"
      onClick={() => document.getElementById("file-upload")?.click()}
    >
      {preview ? (
        <div className="relative w-full h-64">
          <Image
            src={preview}
            alt="CT Scan Preview"
            fill
            className="object-contain"
          />
        </div>
      ) : (
        <>
          <FileImage className="h-12 w-12 text-gray-400 mb-4" />
          <p className="text-sm text-gray-500">
            Click to upload or drag and drop
          </p>
          <p className="text-xs text-gray-400 mt-1">
            CT scan images only (PNG, JPG, JPEG)
          </p>
        </>
      )}
      <input
        id="file-upload"
        type="file"
        className="hidden"
        accept="image/png, image/jpeg, image/jpg"
        onChange={onFileSelect}
        disabled={isUploading}
      />
    </div>
  );
};
