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
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Text",
          "SF Pro Display",
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
