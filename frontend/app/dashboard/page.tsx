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

  const baseUrl = process.env.NEXTAUTH_URL ?? ""; 
  const headersList = await headers();
  const apiPath = "/api/cancer-detection/reports";

  const inferredHost =
    headersList.get("x-forwarded-host") || headersList.get("host") || process.env.VERCEL_URL || "";
  const inferredProto = headersList.get("x-forwarded-proto") || (process.env.NODE_ENV === "development" ? "http" : "https");
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
    console.error("Failed to fetch reports:", err);
    return (
      <div className="container py-10 mx-auto">
        <div className="flex items-center justify-between mb-10">
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <h2 className="text-xl font-medium">Welcome, {session.user.name || "User"}</h2>
        </div>
        <div className="p-4 text-red-600 bg-red-100 border border-red-200 rounded-md">
          Error loading reports. Please try again later.
        </div>
      </div>
    );
  }

  if (!response.ok) {
    console.error(`Failed to fetch reports: ${response.status}`);

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


  const dbReports: DbReport[] = await response.json();


  const reports = dbReports.map((report: DbReport) => ({
    id: report.id,
    imageUrl: report.imageUrl,
    status: report.status,

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
