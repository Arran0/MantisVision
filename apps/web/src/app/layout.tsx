import type { Metadata, Viewport } from "next";
import { Poppins } from "next/font/google";
import "./globals.css";
import { ServiceWorkerRegister } from "@/components/ServiceWorkerRegister";
import { Background } from "@/components/Background";
import { Logo } from "@/components/Logo";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-poppins",
});

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
  themeColor: "#ffffff",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={poppins.variable}>
      <body className="min-h-screen font-sans text-slate-900 antialiased">
        <ServiceWorkerRegister />
        <Background />

        {/* Thick white top bar: logo sits before the title to save space. */}
        <header className="sticky top-0 z-20 w-full border-b border-slate-200/70 bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex h-20 w-full max-w-[1600px] items-center gap-3 px-5 sm:px-8">
            <Logo className="h-11 w-11 shrink-0 drop-shadow-[0_4px_10px_rgba(230,126,48,0.25)]" />
            <span className="text-2xl font-bold tracking-tight sm:text-3xl">
              <span className="mv-text-gradient">Mantis Vision</span>
            </span>
            <span className="ml-auto hidden text-sm text-slate-400 sm:block">
              Google Lens for Seaweed
            </span>
          </div>
        </header>

        {children}
      </body>
    </html>
  );
}
