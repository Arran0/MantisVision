import type { Metadata, Viewport } from "next";
import "./globals.css";
import { ServiceWorkerRegister } from "@/components/ServiceWorkerRegister";
import { Background } from "@/components/Background";

export const metadata: Metadata = {
  title: "Mantis Vision",
  description: "Google Lens for Seaweed — photograph a specimen, get a health assessment.",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Mantis Vision",
  },
  other: {
    // Standard PWA install-capable tag; appleWebApp above covers older iOS.
    "mobile-web-app-capable": "yes",
  },
};

export const viewport: Viewport = {
  themeColor: "#1a7ae0",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans text-slate-900 antialiased">
        <ServiceWorkerRegister />
        <Background />
        {children}
      </body>
    </html>
  );
}
