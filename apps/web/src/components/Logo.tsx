// The Mantis Vision mark: an orange eye (vesica) whose iris is the three
// brand dots — orange, ocean blue, seaweed green. Scales crisply at any size.
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 400 400" className={className} role="img" aria-label="Mantis Vision">
      {/* Orange eye outline with a white interior punched out via even-odd fill. */}
      <path
        fill="#e67e30"
        fillRule="evenodd"
        d="M200 110
           C120 110 66 165 40 200
           C66 235 120 290 200 290
           C280 290 334 235 360 200
           C334 165 280 110 200 110 Z
           M200 138
           C234 138 262 166 262 200
           C262 234 234 262 200 262
           C166 262 138 234 138 200
           C138 166 166 138 200 138 Z"
      />
      {/* Three brand dots forming the iris. */}
      <circle cx="200" cy="180" r="18" fill="#e67e30" />
      <circle cx="181" cy="212" r="18" fill="#1f6fc4" />
      <circle cx="219" cy="212" r="18" fill="#1f9e5a" />
    </svg>
  );
}
