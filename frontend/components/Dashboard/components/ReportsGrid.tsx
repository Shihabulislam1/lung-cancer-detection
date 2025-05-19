"use client";

import React, { useEffect, useState } from "react";
import { ReportCard, ReportSummary } from "@/components/Dashboard/components/ReportCard";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";

export interface ReportsGridProps {
  initialReports?: ReportSummary[];
}

export const ReportsGrid: React.FC<ReportsGridProps> = ({ initialReports }) => {
  const [reports, setReports] = useState<ReportSummary[]>(initialReports || []);
  const [isLoading, setIsLoading] = useState<boolean>(!initialReports);
  const { toast } = useToast();

  useEffect(() => {
    if (!initialReports) {
      fetchReports();
    }
  }, [initialReports]);

  const fetchReports = async () => {
    try {
      setIsLoading(true);
      const response = await fetch("/api/cancer-detection/reports");
      
      if (!response.ok) {
        throw new Error("Failed to fetch reports");
      }
      
      const data = await response.json();
      setReports(data);
    } catch (error) {
      console.error("Error fetching reports:", error);
      toast({
        title: "Error",
        description: "Failed to load your reports. Please try again later.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="flex flex-col space-y-3">
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-6 w-1/3" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ))}
      </div>
    );
  }

  if (reports.length === 0) {
    return (
      <div className="py-12 text-center">
        <h3 className="text-lg font-medium text-gray-700">No reports found</h3>
        <p className="text-gray-500 mt-2">
          Upload a CT scan in the Cancer Detector to create your first report.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {reports.map((report) => (
        <ReportCard key={report.id} report={report} />
      ))}
    </div>
  );
};
