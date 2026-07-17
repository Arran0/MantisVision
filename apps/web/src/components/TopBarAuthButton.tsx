"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

// Same slim dewberry pill either way so the top bar never shifts size when
// auth state flips — only the label and the action underneath change.
const authButtonClass =
  "ml-auto shrink-0 rounded-full bg-dewberry-700 px-4 py-2 text-xs font-semibold text-dewberry-50 transition duration-200 hover:bg-dewberry-800 active:scale-95 sm:ml-3 sm:px-5 sm:py-2.5 sm:text-sm";

export function TopBarAuthButton({ signedIn }: { signedIn: boolean }) {
  const router = useRouter();

  if (!signedIn) {
    return (
      <Link href="/admin/login" className={authButtonClass}>
        Sign in
      </Link>
    );
  }

  async function signOut() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/");
    router.refresh();
  }

  return (
    <button type="button" onClick={signOut} className={authButtonClass}>
      Sign out
    </button>
  );
}
