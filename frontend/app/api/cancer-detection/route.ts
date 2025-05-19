import { type NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import type { Session } from "next-auth";
import { authOptions } from "@/lib/auth";
import { db } from "@/lib/db";
import { cancerReports } from "@/lib/db/schema";
import { uploadImage } from "@/lib/cloudinary";
import { eq } from "drizzle-orm";

export async function POST(request: NextRequest) {
  try {
    const session = (await getServerSession(authOptions)) as Session | null;

    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Check file type
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        { error: "File must be an image" },
        { status: 400 }
      );
    }

    // Convert file to buffer
    const buffer = Buffer.from(await file.arrayBuffer());

    // Upload to Cloudinary
    const { url: imageUrl, publicId } = await uploadImage(buffer);

    // Create initial report entry
    const [report] = await db
      .insert(cancerReports)
      .values({
        userId: Number.parseInt((session as Session).user.id),
        imageUrl,
        publicId,
        status: "pending",
      })
      .returning();

    // Send request to ML backend
    try {
      // ML API integration is commented out as it's not ready yet
      // Will be uncommented when ML API is ready
      /*
      let mlData;
      if (process.env.ML_API_URL) {
        const mlResponse = await fetch(process.env.ML_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ imageUrl }),
        });

        if (!mlResponse.ok) {
          throw new Error(`ML API responded with status: ${mlResponse.status}`);
        }

        mlData = await mlResponse.json();
      } else {
      */

      // Always generate mock data for now until ML API is ready
      console.log("Using mock data for cancer detection.");

      // Simplified mock data - using only 3 classes (cancer, no cancer, inconclusive)
      const randomClasses = 3; // Only using 3 classes for frontend display

      // Create mock raw output data
      const rawOutput = [[Math.random(), Math.random(), Math.random()]];

      // Create normalized probability distribution
      const unnormalizedProbs = [Math.random(), Math.random(), Math.random()];
      const sum = unnormalizedProbs.reduce((a, b) => a + b, 0);
      const probabilities = [unnormalizedProbs.map((p) => p / sum)];

      // Pick a class based on highest probability
      const predictedClassIndex = unnormalizedProbs.indexOf(
        Math.max(...unnormalizedProbs)
      );

      const mlData = {
        rawOutput,
        probabilities,
        predictedClassIndex,
      };
      // }

      // Update report with ML results
      const [updatedReport] = await db
        .update(cancerReports)
        .set({
          status: "completed",
          rawOutput: mlData.rawOutput,
          probabilities: mlData.probabilities,
          predictedClassIndex: mlData.predictedClassIndex,
          updatedAt: new Date(),
        })
        .where(eq(cancerReports.id, report.id))
        .returning();

      return NextResponse.json(updatedReport, { status: 200 });
    } catch (error) {
      console.error("ML processing error:", error);

      // Update report status to failed
      await db
        .update(cancerReports)
        .set({
          status: "failed",
          updatedAt: new Date(),
        })
        .where(eq(cancerReports.id, report.id));

      return NextResponse.json(
        { error: "Error processing image with ML model", reportId: report.id },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Error uploading image" },
      { status: 500 }
    );
  }
}
