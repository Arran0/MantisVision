"use client";

// Shared admin-side visual primitives — flat, neutral (zinc), near-static
// (no idle motion; only a brief enter/exit via Framer Motion where content
// actually appears/disappears, e.g. an expanding row or a modal). Corners are
// square-to-crisp (rounded-none surfaces, a 2px rounded-sm on interactive
// controls) rather than the soft pill/rounded look — the dashboard is a dense
// tool, not a toy. Distinct on purpose from the public-facing mv-card/mv-btn-*
// system (globals.css), which stays playful/translucent for the predict flow.

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from "react";

export const inputClass =
  "w-full rounded-sm border border-zinc-300 bg-white px-3 py-2.5 text-sm text-zinc-900 transition-colors focus:border-zinc-900 focus:outline-none disabled:bg-zinc-50 disabled:text-zinc-400";

export const labelClass = "mb-1.5 block text-[10px] font-bold uppercase tracking-widest text-zinc-500";

export const sectionHeadingClass =
  "mb-3 mt-1 border-b border-zinc-200 pb-1 text-[10px] font-bold uppercase tracking-widest text-dewberry-600";

export function AdminField({
  label,
  children,
  className = "",
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`block ${className}`}>
      <span className={labelClass}>{label}</span>
      {children}
    </label>
  );
}

export function AdminInput(props: InputHTMLAttributes<HTMLInputElement>) {
  const { className = "", ...rest } = props;
  return <input className={`${inputClass} ${className}`} {...rest} />;
}

export function AdminSelect(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  const { className = "", ...rest } = props;
  return <select className={`${inputClass} ${className}`} {...rest} />;
}

export function AdminTextarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const { className = "", ...rest } = props;
  return <textarea className={`${inputClass} ${className}`} {...rest} />;
}

export function AdminCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-none border border-zinc-200 bg-white shadow-sm ${className}`}>{children}</div>;
}

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-zinc-900 text-white hover:bg-zinc-800",
  secondary: "border border-zinc-300 text-zinc-900 hover:bg-zinc-100",
  danger: "border border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100",
  ghost: "text-dewberry-700 hover:bg-dewberry-50",
};

export function AdminButton({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-sm px-4 py-2.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${BUTTON_VARIANTS[variant]} ${className}`}
      {...props}
    />
  );
}

type BadgeTone = "zinc" | "dewberry" | "seaweed" | "amber" | "rose";

const BADGE_TONES: Record<BadgeTone, string> = {
  zinc: "bg-zinc-100 text-zinc-700 border-zinc-200",
  dewberry: "bg-dewberry-50 text-dewberry-700 border-dewberry-200",
  seaweed: "bg-seaweed-50 text-seaweed-700 border-seaweed-200",
  amber: "bg-amber-50 text-amber-700 border-amber-200",
  rose: "bg-rose-50 text-rose-700 border-rose-200",
};

export function AdminBadge({ children, tone = "zinc" }: { children: ReactNode; tone?: BadgeTone }) {
  return (
    <span
      className={`inline-flex items-center rounded-sm border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${BADGE_TONES[tone]}`}
    >
      {children}
    </span>
  );
}

// A flat metric tile for the Home dashboard — big number, small caption, an
// optional sub-line for context. Square corners, hairline border, no gradient.
export function AdminStat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="rounded-none border border-zinc-200 bg-white p-5">
      <p className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">{label}</p>
      <p className="mt-2 text-3xl font-bold tabular-nums tracking-tight text-zinc-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-zinc-500">{hint}</p>}
    </div>
  );
}

// A single dark masthead atop each admin page — icon, title, subtitle, and an
// optional actions slot on the right. Static (no gradient/parallax/motion).
export function AdminPageHeader({
  icon,
  title,
  subtitle,
  actions,
}: {
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-none bg-dewberry-900 px-5 py-4">
      <div className="flex items-center gap-3">
        {icon && <div className="flex h-9 w-9 items-center justify-center rounded-sm bg-white/10 text-dewberry-300">{icon}</div>}
        <div>
          <h1 className="text-base font-bold text-white">{title}</h1>
          {subtitle && <p className="text-xs text-zinc-400">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
