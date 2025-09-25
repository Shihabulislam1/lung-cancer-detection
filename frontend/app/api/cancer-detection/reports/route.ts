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

// Persist a new cancer detection report.
// Expected JSON body: { imageUrl: string, probabilities?: number[][], predictedClassIndex?: number, rawOutput?: number[][] }
export async function POST(req: Request) {
  try {
    const session = (await getServerSession(authOptions)) as Session | null;
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json().catch(() => null);
    if (!body) {
      return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
    }

    const { imageUrl, probabilities, predictedClassIndex, rawOutput } = body;

    if (!imageUrl || typeof imageUrl !== "string") {
      return NextResponse.json(
        { error: "imageUrl is required" },
        { status: 400 }
      );
    }

    // Basic shape checks (optional / lenient)
    const probValue = Array.isArray(probabilities) ? probabilities : undefined;
    const rawValue = Array.isArray(rawOutput) ? rawOutput : undefined;
    const pciValue =
      typeof predictedClassIndex === "number" ? predictedClassIndex : undefined;

    const userIdNum = Number.parseInt((session as Session).user.id);

    const [inserted] = await db
      .insert(cancerReports)
      .values({
        userId: userIdNum,
        imageUrl, // Could be data URL or remote storage URL
        probabilities: probValue,
        rawOutput: rawValue,
        predictedClassIndex: pciValue,
        status: "completed",
      })
      .returning();

    return NextResponse.json(inserted, { status: 201 });
  } catch (error) {
    console.error("Error creating report:", error);
    return NextResponse.json(
      { error: "Error creating report" },
      { status: 500 }
    );
  }
}
