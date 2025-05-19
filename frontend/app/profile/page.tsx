import { getServerSession } from "next-auth/next"
import { redirect } from "next/navigation"
import { authOptions } from "@/lib/auth"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default async function ProfilePage() {
  const session = await getServerSession(authOptions)

  if (!session) {
    redirect("/auth/login")
  }

  return (
    <div className="container py-10 mx-auto">
      <h1 className="text-3xl font-bold mb-6">Profile</h1>
      <Card>
        <CardHeader>
          <CardTitle>User Information</CardTitle>
          <CardDescription>Your personal information</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-2">
            <div className="font-medium">Name</div>
            <div>{session.user.name}</div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="font-medium">Email</div>
            <div>{session.user.email}</div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
