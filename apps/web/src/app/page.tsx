import { UploadCard } from "@/components/UploadCard";
import { Logo } from "@/components/Logo";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-xl flex-col items-center gap-8 px-5 pb-16 pt-12 sm:pt-16">
      <header className="flex flex-col items-center text-center">
        <Logo className="h-16 w-16 rounded-[15px] shadow-lg shadow-ocean-500/20" />
        <h1 className="mt-5 text-4xl font-semibold tracking-tight">
          <span className="mv-text-gradient">Mantis Vision</span>
        </h1>
        <p className="mt-2 text-lg text-slate-500">Google Lens for Seaweed</p>
      </header>

      <UploadCard />
    </main>
  );
}
