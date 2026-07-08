// The Mantis Vision mark: an eye on a squircle carrying the mantis-shrimp
// gradient (orange → ocean blue → seaweed green). Scales crisply at any size.
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} role="img" aria-label="Mantis Vision">
      <defs>
        <linearGradient id="mv-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ff7a1a" />
          <stop offset="0.5" stopColor="#1a7ae0" />
          <stop offset="1" stopColor="#16a34a" />
        </linearGradient>
        <radialGradient id="mv-iris" cx="0.5" cy="0.45" r="0.55">
          <stop offset="0" stopColor="#16a34a" />
          <stop offset="1" stopColor="#1a7ae0" />
        </radialGradient>
      </defs>
      <rect x="0" y="0" width="64" height="64" rx="15" fill="url(#mv-bg)" />
      <ellipse cx="32" cy="32" rx="22.4" ry="13.8" fill="#ffffff" />
      <circle cx="32" cy="32" r="10.6" fill="url(#mv-iris)" stroke="#ff7a1a" strokeWidth="1.2" />
      <circle cx="32" cy="32" r="4.7" fill="#071a3d" />
      <circle cx="29.5" cy="29.5" r="1.9" fill="#ffffff" fillOpacity="0.92" />
    </svg>
  );
}
