import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ResolveGrid",
  description: "Kestrel Softworks internal IT service-management + AI-ops platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
