import { getServerSession } from "next-auth/next";
import { redirect } from "next/navigation";
import { authOptions } from "@/lib/auth";
import CancerDetectorPage from "@/components/Detector/Container/CancerDetectorPage";

export default async function CancerDetectorPageRoute() {
  const session = await getServerSession(authOptions);

  if (!session) {
    redirect("/auth/login");
  }

  return (
    <div className="container mx-auto py-10">
      <h1 className="text-3xl font-bold mb-6 text-center">Cancer Detection Tool</h1>
      <CancerDetectorPage />
    </div>
  );
}
