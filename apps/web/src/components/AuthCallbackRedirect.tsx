"use client";

import { useEffect } from "react";

// Invite/recovery links are built with redirectTo=/member/password-reset,
// but Supabase only honors that when the URL is on the project's Redirect
// URLs allow-list — otherwise it silently swaps in the Site URL instead,
// which is this app's root, and the token (in the URL hash) lands here with
// nowhere to be picked up. Catch that fallback: if a page loads with an auth
// token or error in the hash, forward it to the page that knows what to do
// with it, hash intact.
export function AuthCallbackRedirect() {
  useEffect(() => {
    if (window.location.pathname === "/member/password-reset") return;

    const hash = window.location.hash.replace(/^#/, "");
    if (!hash) return;

    const params = new URLSearchParams(hash);
    const isAuthCallback =
      params.has("access_token") || params.has("error_description") || params.has("token_hash");
    if (isAuthCallback) {
      window.location.replace(`/member/password-reset#${hash}`);
    }
  }, []);

  return null;
}
