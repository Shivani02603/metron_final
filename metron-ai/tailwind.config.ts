import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#00668a",
        "primary-container": "#38bdf8",
        "primary-fixed": "#c4e7ff",
        "primary-fixed-dim": "#7bd0ff",
        secondary: "#006e2f",
        "secondary-container": "#6bff8f",
        "secondary-fixed": "#6bff8f",
        tertiary: "#855300",
        "tertiary-container": "#f1a02b",
        error: "#ba1a1a",
        "error-container": "#ffdad6",
        background: "#f7f9fb",
        surface: "#f7f9fb",
        "surface-bright": "#f7f9fb",
        "surface-dim": "#d8dadc",
        "surface-container-lowest": "#ffffff",
        "surface-container-low": "#f2f4f6",
        "surface-container": "#eceef0",
        "surface-container-high": "#e6e8ea",
        "surface-container-highest": "#e0e3e5",
        "surface-variant": "#e0e3e5",
        "on-surface": "#191c1e",
        "on-surface-variant": "#3e484f",
        "on-primary": "#ffffff",
        "on-primary-container": "#004965",
        "on-secondary": "#ffffff",
        "on-secondary-container": "#007432",
        "on-background": "#191c1e",
        "on-error": "#ffffff",
        outline: "#6e7980",
        "outline-variant": "#bdc8d1",
        "inverse-surface": "#2d3133",
        "inverse-on-surface": "#eff1f3",
        "inverse-primary": "#7bd0ff",
      },
      fontFamily: {
        headline: ["Manrope", "sans-serif"],
        body: ["Inter", "sans-serif"],
        label: ["Inter", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "0.25rem",
        lg: "0.5rem",
        xl: "0.75rem",
        "2xl": "1rem",
        "3xl": "1.5rem",
        full: "9999px",
      },
      boxShadow: {
        card: "0 12px 32px -4px rgba(0, 102, 138, 0.06)",
        "card-lg": "0 20px 48px -12px rgba(56, 189, 248, 0.15)",
        glow: "0 0 8px rgba(107, 255, 143, 0.8)",
      },
      backgroundImage: {
        "primary-gradient": "linear-gradient(135deg, #00668a 0%, #38bdf8 100%)",
        "brand-panel":
          "linear-gradient(135deg, #001e2c 0%, #004c69 50%, #00668a 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
