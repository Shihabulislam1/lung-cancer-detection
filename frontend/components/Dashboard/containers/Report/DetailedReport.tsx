import React from "react";
import { format } from "date-fns";
import Image from "next/image";
import Link from "next/link";
import { ChevronLeft } from "lucide-react";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

// Detailed report model
export interface DetailedReport {
  id: number;
  imageUrl: string;
  status: "pending" | "completed" | "failed";
  rawOutput?: number[][];
  probabilities?: number[][];
  predictedClassIndex?: number;
  createdAt: string;
  updatedAt: string;
}

interface DetailedReportProps {
  report: DetailedReport;
}

const DetailedReport: React.FC<DetailedReportProps> = ({ report }) => {
  const getResultText = (index?: number): string => {
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
  };

  const getResultColor = (index?: number): string => {
    if (index === undefined) return "text-gray-600";

    switch (index) {
      case 0:
        return "text-red-600";
      case 1:
        return "text-green-600";
      case 2:
        return "text-amber-500";
      default:
        return "text-gray-600";
    }
  };
  
  const getStatusBadge = () => {
    const statusColors = {
      pending: "bg-blue-100 text-blue-800",
      completed: "bg-green-100 text-green-800",
      failed: "bg-red-100 text-red-800",
    };

    return (
      <Badge 
        className={statusColors[report.status]} 
        variant="outline"
      >
        {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
      </Badge>
    );
  };

  const formatDate = (dateString: string) => {
    return format(new Date(dateString), "MMMM d, yyyy 'at' h:mm a");
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-2">
        <Link href="/dashboard">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </Button>
        </Link>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-2xl font-bold">CT Scan Report #{report.id}</h2>
              <p className="text-muted-foreground">
                Uploaded on {formatDate(report.createdAt)}
              </p>
            </div>
            {getStatusBadge()}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="relative aspect-square bg-muted rounded-md overflow-hidden">
              <Image
                src={report.imageUrl}
                alt="CT Scan"
                fill
                className="object-contain"
              />
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-1">Analysis Results</h3>
                <Separator className="my-2" />
                
                {report.status === "pending" ? (
                  <div className="py-4">
                    <p className="text-blue-600 font-medium">Analysis in progress...</p>
                    <p className="text-sm text-gray-500 mt-2">
                      Your CT scan is currently being analyzed. This may take a few minutes.
                    </p>
                  </div>
                ) : report.status === "failed" ? (
                  <div className="py-4">
                    <p className="text-red-600 font-medium">Analysis failed</p>
                    <p className="text-sm text-gray-500 mt-2">
                      There was a problem analyzing your CT scan. Please try uploading it again.
                    </p>
                  </div>
                ) : (
                  <>
                    <h4 className={`text-xl font-bold ${getResultColor(report.predictedClassIndex)}`}>
                      {getResultText(report.predictedClassIndex)}
                    </h4>

                    {report.probabilities && (
                      <div className="mt-6 space-y-4">
                        <h5 className="font-medium text-sm">Probability Breakdown:</h5>
                        <div className="space-y-3">
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Cancer:</span>
                              <span className="font-medium">
                                {(report.probabilities[0][0] * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress
                              value={report.probabilities[0][0] * 100}
                              className="h-2"
                              indicatorClassName="bg-red-500"
                            />
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>No Cancer:</span>
                              <span className="font-medium">
                                {(report.probabilities[0][1] * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress
                              value={report.probabilities[0][1] * 100}
                              className="h-2"
                              indicatorClassName="bg-green-500"
                            />
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Inconclusive:</span>
                              <span className="font-medium">
                                {(report.probabilities[0][2] * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress
                              value={report.probabilities[0][2] * 100}
                              className="h-2"
                              indicatorClassName="bg-amber-500"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
                
                <div className="mt-8 p-4 bg-yellow-50 border border-yellow-100 rounded-md">
                  <p className="text-sm text-gray-700">
                    <strong>Important:</strong> This analysis is provided as a screening tool and should not 
                    replace professional medical advice. Please consult with a healthcare provider for proper diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DetailedReport;
