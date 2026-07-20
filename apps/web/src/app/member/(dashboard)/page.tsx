import { redirect } from "next/navigation";

// Home is the first tab every dashboard user lands on.
export default function AdminIndexPage() {
  redirect("/member/home");
}
