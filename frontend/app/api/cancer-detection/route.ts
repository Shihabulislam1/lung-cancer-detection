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
      const ML_API_ENDPOINT =
        process.env.ML_API_URL ||
        process.env.NEXT_PUBLIC_ML_API_URL ||
        "https://shhabul-cancer-api.hf.space/predict"; // fallback for local dev

      // Hugging Face private Spaces (or gated endpoints) require a Bearer token.
      // Keep this server-side only (do NOT expose tokens via NEXT_PUBLIC_*).
      const bearerToken =  process.env.BEARER_TOKEN || "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";

      // Build multipart form data with the original uploaded image file
      const mlForm = new FormData();
      mlForm.append("file", file, file.name);

      const mlResp = await fetch(ML_API_ENDPOINT, {
        method: "POST",
        body: mlForm,
        headers: {
          // Don't set Content-Type - let browser set it with boundary for multipart
          ...(bearerToken ? { Authorization: `Bearer ${bearerToken}` } : {}),
        },
      });

      if (!mlResp.ok) {
        throw new Error(
          `ML API responded with status: ${mlResp.status} (${ML_API_ENDPOINT})`
        );
      }

      const mlJson = await mlResp.json();
      // Expected shape: { filename, prediction: { label, confidence, probabilities: {label: prob} } }
      const prediction = mlJson?.prediction;
      const probsMap: Record<string, number> = prediction?.probabilities || {};

      // Map model labels to internal 3-category schema:
      // 0: Malignant, 1: Normal, 2: Benign
      // Model labels can vary (e.g., "Malignant cases" / "Normal cases" / "Bengin cases").
      const malignantProb = probsMap["Malignant cases"] ?? probsMap["Malignant"] ?? 0;
      const normalProb = probsMap["Normal cases"] ?? probsMap["Normal"] ?? 0;
      const benignProb =
        probsMap["Bengin cases"] ?? probsMap["Benign cases"] ?? probsMap["Benign"] ?? 0;

      const ordered = [malignantProb, normalProb, benignProb];
      const predictedClassIndex = ordered.indexOf(Math.max(...ordered));

      const mlData = {
        probabilities: [ordered],
        predictedClassIndex,
      };

      // Update report with ML results
      const [updatedReport] = await db
        .update(cancerReports)
        .set({
          status: "completed",
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
