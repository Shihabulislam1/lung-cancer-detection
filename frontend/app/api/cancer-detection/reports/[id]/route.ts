import { type NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import type { Session } from "next-auth";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";
import { cancerReports } from "@/lib/db/schema";
import { and, eq } from "drizzle-orm";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = (await getServerSession(authOptions)) as Session | null;

    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Need to await params since it's a dynamic API in Next.js
    const { id } = await params;
    const reportId = Number.parseInt(id);

    if (isNaN(reportId)) {
      return NextResponse.json({ error: "Invalid report ID" }, { status: 400 });
    }

    const report = await db.query.cancerReports.findFirst({
      where: and(
        eq(cancerReports.id, reportId),
        eq(cancerReports.userId, Number.parseInt((session as Session).user.id))
      ),
    });

    if (!report) {
      return NextResponse.json({ error: "Report not found" }, { status: 404 });
    }

    return NextResponse.json(report, { status: 200 });
  } catch (error) {
    console.error("Error fetching report:", error);
    return NextResponse.json(
      { error: "Error fetching report" },
      { status: 500 }
    );
  }
}
