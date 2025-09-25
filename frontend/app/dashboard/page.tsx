import { getServerSession } from "next-auth/next";
import { redirect } from "next/navigation";
import { headers } from "next/headers";
import type { Session } from "next-auth";
import { authOptions } from "@/lib/auth";
import AllReports from "@/components/Dashboard/containers/AllReports/AllReports";

// Type for the database report structure
type DbReport = {
  id: number;
  imageUrl: string;
  status: "pending" | "completed" | "failed";
  predictedClassIndex?: number;
  createdAt: string;
};

export default async function DashboardPage() {
  const session = (await getServerSession(authOptions)) as Session | null;

  if (!session) {
    redirect("/auth/login");
  }

  // Fetch user reports from the API using server component native fetch
  // Make sure to pass cookies for auth to work properly
  const baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3000";
  const headersList = await headers();
  const response = await fetch(`${baseUrl}/api/cancer-detection/reports`, {
    cache: "no-store",
    headers: {
      Cookie: headersList.get("cookie") || "",
    },
  });

  if (!response.ok) {
    console.error(`Failed to fetch reports: ${response.status}`);
    // Return valid jsx instead of an object
    return (
      <div className="container py-10 mx-auto">
        <div className="flex items-center justify-between mb-10">
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <h2 className="text-xl font-medium">
            Welcome, {session.user.name || "User"}
          </h2>
        </div>
        <div className="p-4 text-red-600 bg-red-100 border border-red-200 rounded-md">
          Error loading reports. Please try again later.
        </div>
      </div>
    );
  }

  // Get the reports from the API response
  const dbReports: DbReport[] = await response.json();

  // Transform database reports to match ReportSummary type
  const reports = dbReports.map((report: DbReport) => ({
    id: report.id,
    imageUrl: report.imageUrl,
    status: report.status,
    // Preserve 0 (cancer) rather than dropping via falsy fallback
    predictedClassIndex:
      typeof report.predictedClassIndex === "number"
        ? report.predictedClassIndex
        : undefined,
    createdAt: report.createdAt,
  }));

  return (
    <div className="container py-10 mx-auto">
      <div className="flex items-center justify-between mb-10">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <h2 className="text-xl font-medium">
          Welcome, {session.user.name || "User"}
        </h2>
      </div>

      <AllReports initialReports={reports} />
    </div>
  );
}
