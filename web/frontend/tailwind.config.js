/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0A0E14",
        surface: "#10151F",
        raised: "#161C29",
        hairline: "#232B3A",
        ink: "#E4EAF2",
        muted: "#8592A6",
        safe: "#2DD4A7",
        alert: "#FF5C5C",
        warn: "#F5A623",
        info: "#5B8DEF",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};