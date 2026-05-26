import type { Metadata } from "next";
import { Providers } from "@/components/providers";
import { Sidebar } from "@/components/sidebar/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Studio",
  description: "Local RAG with LM Studio + pgvector",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" className="dark">
      <body>
        <Providers>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <main className="flex-1 min-w-0">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
