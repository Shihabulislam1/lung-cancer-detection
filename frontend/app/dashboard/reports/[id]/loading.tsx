import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function Loading() {
  return (
    <div className="container py-10 mx-auto">
      <div className="space-y-6">
        <div className="flex items-center space-x-2">
          <Skeleton className="h-10 w-32" />
        </div>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex justify-between items-start">
              <div className="space-y-2">
                <Skeleton className="h-8 w-64" />
                <Skeleton className="h-4 w-40" />
              </div>
              <Skeleton className="h-6 w-24" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <Skeleton className="h-80 w-full" />
              <div className="space-y-6">
                <div>
                  <Skeleton className="h-6 w-1/2 mb-3" />
                  <Skeleton className="h-px w-full my-2" />

                  <div className="py-4">
                    <Skeleton className="h-8 w-3/4 mb-3" />
                    <div className="mt-6 space-y-4">
                      <Skeleton className="h-4 w-1/3" />
                      <div className="space-y-3">
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Skeleton className="h-4 w-16" />
                            <Skeleton className="h-4 w-8" />
                          </div>
                          <Skeleton className="h-2 w-full" />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Skeleton className="h-4 w-16" />
                            <Skeleton className="h-4 w-8" />
                          </div>
                          <Skeleton className="h-2 w-full" />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Skeleton className="h-4 w-16" />
                            <Skeleton className="h-4 w-8" />
                          </div>
                          <Skeleton className="h-2 w-full" />
                        </div>
                      </div>
                    </div>
                  </div>

                  <Skeleton className="h-24 w-full mt-8" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
