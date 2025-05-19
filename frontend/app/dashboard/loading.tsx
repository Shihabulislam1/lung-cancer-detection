import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return (
    <div className="container py-10 mx-auto">
      <div className="flex items-center justify-between mb-10">
        <Skeleton className="h-10 w-40" />
        <Skeleton className="h-6 w-56" />
      </div>

      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <Skeleton className="h-8 w-56" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="flex flex-col space-y-3">
              <Skeleton className="h-40 w-full" />
              <div className="px-4 py-2">
                <Skeleton className="h-6 w-1/3 mb-2" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-8 w-full mt-4" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
