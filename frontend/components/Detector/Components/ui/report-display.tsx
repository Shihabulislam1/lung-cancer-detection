"use client";

import React from "react";
import Image from "next/image";
import { CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { Report } from "../models/report.model";
import { CancerDetectionService } from "../services/cancer-detection.service";

interface ReportDisplayProps {
  report: Report;
}

export const ReportDisplay: React.FC<ReportDisplayProps> = ({ report }) => {
  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div className="relative w-full h-64 md:h-full">
        <Image
          src={report.imageUrl || "/placeholder.svg"}
          alt="CT Scan"
          fill
          className="object-contain rounded-md"
        />
      </div>

      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium">Analysis Results</h3>
          <p className="text-sm text-gray-500">
            Scan uploaded on {new Date(report.createdAt).toLocaleString()}
          </p>
        </div>

        {report.status === "pending" ? (
          <div className="flex items-center text-amber-500">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Analysis in progress...
          </div>
        ) : report.status === "failed" ? (
          <div className="flex items-center text-red-500">
            <AlertCircle className="mr-2 h-5 w-5" />
            Analysis failed. Please try again.
          </div>
        ) : (
          <>
            <div className="space-y-2">
              <div className="flex items-center">
                <CheckCircle className="mr-2 h-5 w-5 text-green-500" />
                <span className="font-medium">Analysis Complete</span>
              </div>
              <h4
                className={cn(
                  "text-xl font-bold",
                  CancerDetectionService.getResultColor(
                    report.predictedClassIndex
                  )
                )}
              >
                {CancerDetectionService.getResultMessage(
                  report.predictedClassIndex
                )}
              </h4>
            </div>

            {report.probabilities && (
              <ProbabilityChart probabilities={report.probabilities} />
            )}

            <div className="pt-4">
              <p className="text-sm text-gray-500">
                Note: This analysis is provided as a screening tool and should
                not replace professional medical advice. Please consult with a
                healthcare provider for proper diagnosis.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

interface ProbabilityChartProps {
  probabilities: number[][];
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  probabilities,
}) => {
  return (
    <div className="space-y-3">
      <h4 className="font-medium text-sm">Probability Breakdown:</h4>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Cancer:</span>
          <span className="font-medium">
            {(probabilities[0][0] * 100).toFixed(1)}%
          </span>
        </div>
        <Progress
          value={probabilities[0][0] * 100}
          className="h-2"
          indicatorClassName="bg-red-500"
        />
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>No Cancer:</span>
          <span className="font-medium">
            {(probabilities[0][1] * 100).toFixed(1)}%
          </span>
        </div>
        <Progress
          value={probabilities[0][1] * 100}
          className="h-2"
          indicatorClassName="bg-green-500"
        />
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Inconclusive:</span>
          <span className="font-medium">
            {(probabilities[0][2] * 100).toFixed(1)}%
          </span>
        </div>
        <Progress
          value={probabilities[0][2] * 100}
          className="h-2"
          indicatorClassName="bg-amber-500"
        />
      </div>
    </div>
  );
};
