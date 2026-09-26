import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Nomo — tri-domain HW-NAS",
  description: "Live neuromorphic architecture search telemetry",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-black text-neutral-100 antialiased">{children}</body>
    </html>
  );
}
