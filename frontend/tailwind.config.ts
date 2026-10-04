import type { Config } from "tailwindcss";

// Nomo v4 tokens. Colour carries meaning only: one hue per computing style, one for crossings.
export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#1D2433", soft: "#3A4356", muted: "#5B6475", faint: "#8B93A3" },
        paper: "#EEF1F5",
        panel: "#FFFFFF",
        line: { DEFAULT: "#D5DBE3", strong: "#AEB7C4" },
        ann: { DEFAULT: "#2F5BEA", tint: "#E5ECFD", ink: "#1C3FB0" },
        snn: { DEFAULT: "#E39B17", tint: "#FDF1DA", ink: "#8A5600" },
        sym: { DEFAULT: "#0F8A6C", tint: "#DDF3EC", ink: "#0A6650" },
        cross: { DEFAULT: "#C8385A", tint: "#FBE4EA" },
      },
      fontFamily: {
        sans: ["'Atkinson Hyperlegible'", "system-ui", "-apple-system", "'Segoe UI'", "sans-serif"],
        code: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: { "2xs": ["0.6875rem", { lineHeight: "1rem" }] },
    },
  },
  plugins: [],
} satisfies Config;
