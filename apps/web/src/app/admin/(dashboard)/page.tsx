import { redirect } from "next/navigation";

// No standalone overview page — Dataset is the first real tab admins land on.
export default function AdminIndexPage() {
  redirect("/admin/dataset");
}
