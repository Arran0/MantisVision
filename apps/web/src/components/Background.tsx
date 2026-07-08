// Fixed, full-viewport mantis-shrimp gradient backdrop: three soft blurred
// color fields that drift slowly. Sits behind all page content (-z-10) and
// never intercepts input (pointer-events-none).
export function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#f4f6fa]">
      <div className="mv-blob mv-blob-coral" />
      <div className="mv-blob mv-blob-ocean" />
      <div className="mv-blob mv-blob-seaweed" />
      <div className="absolute inset-0 bg-white/40" />
      <div className="mv-vignette absolute inset-0" />
    </div>
  );
}
