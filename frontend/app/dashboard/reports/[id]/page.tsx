import { getServerSession } from "next-auth/next";
import { notFound, redirect } from "next/navigation";
import { headers } from "next/headers";
import type { Session } from "next-auth";
import { authOptions } from "@/lib/auth";
import DetailedReport, {
  DetailedReport as DetailedReportType,
} from "@/components/Dashboard/containers/Report/DetailedReport";

interface ReportPageProps {
  params: {
    id: string;
  };
}

export default async function ReportPage({ params }: ReportPageProps) {
  const session = (await getServerSession(authOptions)) as Session | null;

  if (!session) {
    redirect("/auth/login");
  }

  // Await params to comply with Next.js dynamic API requirements
  const { id } = await params;
  const reportId = Number.parseInt(id);

  // Handle invalid ID
  if (isNaN(reportId)) {
    notFound();
  } // Fetch the report data from the API
  const baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3000";
  const headersList = await headers();
  const response = await fetch(
    `${baseUrl}/api/cancer-detection/reports/${reportId}`,
    {
      cache: "no-store",
      headers: {
        Cookie: headersList.get("cookie") || "",
      },
    }
  );

  // If report not found or doesn't belong to user
  if (!response.ok) {
    if (response.status === 404) {
      notFound();
    }
    throw new Error(`Failed to fetch report: ${response.status}`);
  }

  // Get the report from the API response
  const dbReport = await response.json();

  // Transform the report data to match the expected DetailedReport type
  const formattedReport: DetailedReportType = {
    id: dbReport.id,
    imageUrl: dbReport.imageUrl,
    status: dbReport.status,
    rawOutput: dbReport.rawOutput ?? undefined,
    probabilities: dbReport.probabilities ?? undefined,
    // Preserve 0 (cancer) instead of dropping it via falsy check
    predictedClassIndex:
      typeof dbReport.predictedClassIndex === "number"
        ? dbReport.predictedClassIndex
        : undefined,
    createdAt:
      typeof dbReport.createdAt === "string"
        ? dbReport.createdAt
        : new Date(dbReport.createdAt).toISOString(),
    updatedAt:
      typeof dbReport.updatedAt === "string"
        ? dbReport.updatedAt
        : new Date(dbReport.updatedAt).toISOString(),
  };

  return (
    <div className="container py-10 mx-auto">
      <DetailedReport report={formattedReport} />
    </div>
  );
}
