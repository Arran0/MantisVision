import type { Config } from "tailwindcss";

// Palette drawn from the mantis shrimp: warm orange → ocean blue → seaweed green.
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        coral: {
          400: "#ff8f42",
          500: "#ff7a1a",
          600: "#ea6a0c",
        },
        ocean: {
          50: "#eff6ff",
          100: "#dbeafe",
          400: "#3b93e8",
          500: "#1a7ae0",
          600: "#155fb0",
          700: "#124e90",
        },
        seaweed: {
          50: "#f0fdf6",
          100: "#dcfce9",
          200: "#bbf7d1",
          500: "#16a34a",
          600: "#15803d",
          700: "#166534",
          900: "#14532d",
        },
        // Admin-side accent, drawn from Sherwin-Williams "Dewberry" (SW 6552,
        // #3e385a, anchored at 700) — used alongside the offwhite/zinc-grey
        // admin palette.
        dewberry: {
          50: "#f4eef9",
          100: "#e8dcf1",
          200: "#d1bce3",
          300: "#b092cf",
          400: "#8c6bb0",
          500: "#6b4f8c",
          600: "#4f3c68",
          700: "#3e385a",
          800: "#2a2740",
          900: "#1a1826",
        },
      },
      fontFamily: {
        sans: [
          "var(--font-poppins)",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
