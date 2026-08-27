import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Drury R. Reavill Pathology Database at UC Davis",
  description:
    "Search, filter, and visualize the Drury R. Reavill exotic companion animal pathology database at UC Davis.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
