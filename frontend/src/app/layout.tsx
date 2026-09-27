import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Nomo: design energy-efficient AI for your chip",
  description: "Find the best mix of standard, spiking and physics-based layers for your model and hardware.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-paper font-sans text-ink antialiased">{children}</body>
    </html>
  );
}
