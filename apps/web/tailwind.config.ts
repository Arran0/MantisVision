import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        seaweed: {
          50: "#f0fdf6",
          100: "#dcfce9",
          500: "#16a34a",
          600: "#15803d",
          900: "#14532d",
        },
      },
    },
  },
  plugins: [],
};

export default config;
