import Link from "next/link";
import { FileX } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4">
      <FileX className="h-16 w-16 text-muted-foreground mb-4" />
      <h2 className="text-2xl font-bold">Report Not Found</h2>
      <p className="text-muted-foreground mt-2 mb-6">
        The report you&apos;re looking for doesn&apos;t exist or you don&apos;t
        have permission to view it.
      </p>
      <Link href="/dashboard">
        <Button>Return to Dashboard</Button>
      </Link>
    </div>
  );
}
