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
  // Fetch the report data from the API
  // Build origin from environment or request headers so server fetches work on Vercel
  const baseUrl = process.env.NEXTAUTH_URL ?? "";
  const headersList = await headers();
  const apiPath = `/api/cancer-detection/reports/${reportId}`;
  const inferredHost =
    headersList.get("x-forwarded-host") || headersList.get("host") || process.env.VERCEL_URL || "";
  const inferredProto =
    headersList.get("x-forwarded-proto") || (process.env.NODE_ENV === "development" ? "http" : "https");
  const origin = baseUrl
    ? baseUrl.replace(/\/$/, "")
    : inferredHost
    ? `${inferredProto}://${inferredHost.replace(/\/$/, "")}`
    : "";

  let response: Response;
  try {
    const fetchUrl = origin ? `${origin}${apiPath}` : apiPath;
    response = await fetch(fetchUrl, {
      cache: "no-store",
      headers: {
        Cookie: headersList.get("cookie") || "",
      },
    });
  } catch (err) {
    console.error("Failed to fetch report:", err);
    throw new Error(`Failed to fetch report: ${String(err)}`);
  }

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
}}
