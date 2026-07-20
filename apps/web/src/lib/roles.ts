// Admin-panel account levels. 'admin' has full access; 'contributor' can label
// dataset photos and see their own contribution stats but nothing else;
// 'viewer' is the zero-access default for an auto-provisioned profile.
export type Role = "admin" | "contributor" | "viewer";

// Levels an admin can grant to an invited teammate.
export const ASSIGNABLE_ROLES: Extract<Role, "admin" | "contributor">[] = ["admin", "contributor"];

// Roles allowed into the /member dashboard at all.
export function isDashboardRole(role: string | null | undefined): role is "admin" | "contributor" {
  return role === "admin" || role === "contributor";
}

export function roleLabel(role: Role | string | null | undefined): string {
  switch (role) {
    case "admin":
      return "Admin";
    case "contributor":
      return "Contributor";
    case "viewer":
      return "Viewer";
    default:
      return "—";
  }
}
