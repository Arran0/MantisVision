import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { AdminNav } from "@/components/admin/AdminNav";

// Every authed admin page is wrapped by this layout. /admin/login lives
// outside the (dashboard) route group specifically so it does NOT inherit
// this layout — nesting it here previously caused a redirect loop (an
// unauthenticated visit to /admin/login would hit this layout's redirect
// below, which points right back at /admin/login, forever).
//
// This is where the actual role check happens — middleware.ts only confirms
// a session exists.
export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/admin/login");
  }

  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", user.id)
    .single();

  if (profile?.role !== "admin") {
    return (
      <main className="mx-auto flex max-w-lg flex-col items-center gap-3 px-5 py-24 text-center">
        <h1 className="text-xl font-bold text-slate-900">Not authorized</h1>
        <p className="text-sm text-slate-600">
          Your account ({user.email}) doesn&rsquo;t have admin access. Ask an existing admin to
          promote your account.
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-6 px-5 py-10 sm:px-8">
      <AdminNav email={user.email ?? null} />
      {children}
    </main>
  );
}
