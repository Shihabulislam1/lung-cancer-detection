import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import type { Session } from "next-auth";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";
import { cancerReports } from "@/lib/db/schema";
import { eq } from "drizzle-orm";

export async function GET() {
  try {
    const session = (await getServerSession(authOptions)) as Session | null;

    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const reports = await db.query.cancerReports.findMany({
      where: eq(
        cancerReports.userId,
        Number.parseInt((session as Session).user.id)
      ),
      orderBy: (cancerReports, { desc }) => [desc(cancerReports.createdAt)],
    });

    return NextResponse.json(reports, { status: 200 });
  } catch (error) {
    console.error("Error fetching reports:", error);
    return NextResponse.json(
      { error: "Error fetching reports" },
      { status: 500 }
    );
  }
}
