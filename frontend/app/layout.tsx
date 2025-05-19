import type React from "react";
import "@/app/globals.css";
import { Inter } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import Header from "@/components/Header/header";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider } from "@/components/auth-provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Cancer Detection",
  description: "Lung Cancer Detection of CT Scan Images",
  keywords: "Lung Cancer, CT Scan, Detection, AI, Machine Learning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <AuthProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <div className="min-h-screen flex flex-col">
              <Header />
              <main className="flex-1 px-4 md-px-0">{children}</main>
            </div>
            <Toaster />
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
