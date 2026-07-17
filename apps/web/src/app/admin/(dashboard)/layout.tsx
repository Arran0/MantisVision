import { redirect } from "next/navigation";
import { getDashboardUser } from "@/lib/supabase/page-guards";
import { isDashboardRole } from "@/lib/roles";
import { AdminNav } from "@/components/admin/AdminNav";

// Every authed dashboard page is wrapped by this layout. /admin/login and
// /admin/set-password live OUTSIDE the (dashboard) route group specifically so
// they do NOT inherit this layout — nesting login here previously caused a
// redirect loop (an unauthenticated visit would hit this layout's redirect
// below, which points right back at login, forever), and set-password needs to
// render before a session cookie exists so the client can finish the invite.
//
// This is where the actual role check happens — middleware.ts only confirms a
// session exists. Both admins and contributors reach the dashboard; the nav
// and per-page guards decide what each level can open.
export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const user = await getDashboardUser();

  if (!user) {
    redirect("/admin/login");
  }

  if (!isDashboardRole(user.role)) {
    return (
      <div className="min-h-screen bg-zinc-100">
        <main className="mx-auto flex max-w-lg flex-col items-center gap-3 px-5 py-24 text-center">
          <h1 className="text-xl font-bold text-zinc-900">Not authorized</h1>
          <p className="text-sm text-zinc-600">
            Your account ({user.email}) doesn&rsquo;t have dashboard access. Ask an existing admin to
            grant your account a level.
          </p>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-100">
      <main className="mx-auto flex w-[95%] max-w-none flex-col gap-5 py-8">
        <AdminNav email={user.email} role={user.role} />
        {children}
      </main>
    </div>
  );
}
