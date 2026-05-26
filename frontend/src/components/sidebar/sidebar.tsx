"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";
import { ChatsList } from "./chats-list";
import { SourcesPanel } from "./sources-panel";
import { api } from "@/lib/api";

export function Sidebar() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  return (
    <aside className="flex flex-col h-full border-r border-border bg-card/40 w-72 min-w-[18rem]">
      <header className="px-4 py-3 border-b border-border">
        <h1 className="text-sm font-semibold tracking-tight">RAG Studio</h1>
        <p className="text-[11px] text-muted-foreground">LM Studio + pgvector</p>
      </header>

      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex-[2] min-h-0">
          <ChatsList />
        </div>
        <div className="flex-[3] min-h-0">
          <SourcesPanel />
        </div>
      </div>

      <footer className="px-3 py-2 border-t border-border flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Activity className="h-3 w-3" />
          {healthQuery.data ? `${healthQuery.data.chunks_in_db} чанков` : "…"}
        </span>
        {healthQuery.isError && <span className="text-amber-400">backend offline</span>}
      </footer>
    </aside>
  );
}
