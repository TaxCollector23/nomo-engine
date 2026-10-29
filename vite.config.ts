import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "/lab/" mounts the built Lab under the preview gateway path (public/lab in the
// managed Next shell). Local `npm run dev` is unaffected (base only affects builds;
// set to "/" there if you serve from a domain root).
export default defineConfig(({ mode }) => ({
  base: mode === "production" ? "/lab/" : "/",
  plugins: [react()],
}));
