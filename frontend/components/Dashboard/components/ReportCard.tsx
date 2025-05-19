"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { format } from "date-fns";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Eye } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ReportSummary {
  id: number;
  imageUrl: string;
  status: "pending" | "completed" | "failed";
  predictedClassIndex?: number;
  createdAt: string;
}

interface ReportCardProps {
  report: ReportSummary;
}

export const ReportCard: React.FC<ReportCardProps> = ({ report }) => {
  const router = useRouter();
  
  const getStatusColor = (status: string): string => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800";
      case "pending":
        return "bg-blue-100 text-blue-800";
      case "failed":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getResultText = (index?: number): string => {
    if (index === undefined || report.status !== "completed") return "No results";
    
    switch (index) {
      case 0:
        return "Cancer detected";
      case 1:
        return "No cancer detected";
      case 2:
        return "Inconclusive";
      default:
        return "Unknown result";
    }
  };
  
  const getResultColor = (index?: number): string => {
    if (index === undefined || report.status !== "completed") return "text-gray-600";
    
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

  const viewReport = () => {
    router.push(`/dashboard/reports/${report.id}`);
  };

  return (
    <Card className="overflow-hidden">
      <div 
        className="h-40 bg-center bg-cover" 
        style={{ backgroundImage: `url(${report.imageUrl})` }}
      />
      <CardContent className="pt-6">
        <div className="flex justify-between items-start mb-2">
          <Badge className={getStatusColor(report.status)} variant="outline">
            {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
          </Badge>
          <span className="text-sm text-gray-500">
            {format(new Date(report.createdAt), "MMM d, yyyy")}
          </span>
        </div>
        <h3 className={cn("font-medium mt-2", getResultColor(report.predictedClassIndex))}>
          {getResultText(report.predictedClassIndex)}
        </h3>
      </CardContent>
      <CardFooter className="border-t bg-muted/30 pt-3 pb-3">
        <Button onClick={viewReport} variant="outline" className="w-full" size="sm">
          <Eye className="mr-1 h-4 w-4" />
          View Details
        </Button>
      </CardFooter>
    </Card>
  );
};
