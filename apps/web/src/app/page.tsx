import { UploadCard } from "@/components/UploadCard";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center gap-8 px-4 py-12">
      <header className="text-center">
        <h1 className="text-3xl font-bold text-seaweed-900">Mantis Vision</h1>
        <p className="mt-2 text-slate-600">Google Lens for Seaweed</p>
      </header>

      <UploadCard />
    </main>
  );
}
