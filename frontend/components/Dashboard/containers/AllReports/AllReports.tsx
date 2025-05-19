import React from "react";
import { ReportsGrid, ReportsGridProps } from "@/components/Dashboard/components/ReportsGrid";

interface AllReportsProps extends ReportsGridProps {}

/**
 * AllReports - Container component for displaying all reports
 */
const AllReports: React.FC<AllReportsProps> = ({ initialReports }) => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Your CT Scan Reports</h2>
      </div>
      
      <ReportsGrid initialReports={initialReports} />
    </div>
  );
};

export default AllReports;