import { UploadCard } from "@/components/UploadCard";
import { Logo } from "@/components/Logo";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-[1400px] flex-col items-center gap-10 px-6 pb-20 pt-12 sm:px-10 sm:pt-16 lg:px-16">
      <header className="flex flex-col items-center text-center">
        <Logo className="h-20 w-20 drop-shadow-[0_6px_16px_rgba(230,126,48,0.25)]" />
        <h1 className="mt-5 text-4xl font-semibold tracking-tight sm:text-5xl">
          <span className="mv-text-gradient">Mantis Vision</span>
        </h1>
        <p className="mt-2 text-lg text-slate-500">Google Lens for Seaweed</p>
      </header>

      <UploadCard />
    </main>
  );
}
