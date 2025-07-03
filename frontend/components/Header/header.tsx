"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useSession, signOut } from "next-auth/react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { User } from "lucide-react";

export default function Header() {
  const pathname = usePathname();
  const { status } = useSession();
  const isLoading = status === "loading";
  const isAuthenticated = status === "authenticated";

  return (
    <header className="border-b">
      <div className="container mx-auto flex h-16 items-center justify-between py-4 px-4 md:px-0">
        <div className="flex gap-6 md:gap-10 items-center justify-center">
          <Link href="/" className="flex items-center space-x-2">
            <span className="font-bold text-2xl inline-block">Cancer</span>
          </Link>
          <nav className="hidden gap-6 md:flex">
            <Link
              href="/"
              className={`text-sm font-medium transition-colors hover:text-primary ${
                pathname === "/" ? "text-foreground" : "text-foreground/60"
              }`}
            >
              Home
            </Link>
            {isAuthenticated && (
              <>
                <Link
                  href="/dashboard"
                  className={`text-sm font-medium transition-colors hover:text-primary ${
                    pathname === "/dashboard"
                      ? "text-foreground"
                      : "text-foreground/60"
                  }`}
                >
                  Dashboard
                </Link>
                <Link
                  href="/cancer-detector"
                  className={`text-sm font-medium transition-colors hover:text-primary ${
                    pathname === "/cancer-detector"
                      ? "text-foreground"
                      : "text-foreground/60"
                  }`}
                >
                  Cancer Detector
                </Link>
              </>
            )}
          </nav>
        </div>
        <div className="flex items-center gap-2">
          {isLoading ? (
            <div className="h-8 w-8 animate-pulse rounded-full bg-muted"></div>
          ) : isAuthenticated ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="rounded-full">
                  <User className="h-5 w-5" />
                  <span className="sr-only">User menu</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>
                  <Link href="/profile" className="w-full">
                    Profile
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <Link href="/dashboard" className="w-full">
                    Dashboard
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <Link href="/cancer-detector" className="w-full">
                    Cancer Detector
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => signOut()}>
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <>
              <Button variant="ghost" asChild>
                <Link href="/auth/login">Sign In</Link>
              </Button>
              <Button asChild>
                <Link href="/auth/register">Sign Up</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
