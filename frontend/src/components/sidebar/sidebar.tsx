"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Activity, Sparkles } from "lucide-react";
import { ChatsList } from "./chats-list";
import { SourcesPanel } from "./sources-panel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
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

      <Tabs defaultValue="chats" className="flex-1 flex flex-col min-h-0">
        <TabsList className="grid grid-cols-2 mx-3 mt-2">
          <TabsTrigger value="chats">Чаты</TabsTrigger>
          <TabsTrigger value="agent">Agent</TabsTrigger>
        </TabsList>

        <TabsContent value="chats" className="flex-1 min-h-0 flex flex-col mt-0">
          <div className="flex-[2] min-h-0">
            <ChatsList />
          </div>
          <div className="flex-[3] min-h-0">
            <SourcesPanel />
          </div>
        </TabsContent>

        <TabsContent value="agent" className="flex-1 min-h-0 mt-0">
          <div className="flex flex-col items-center justify-center h-full p-4 gap-3 text-center">
            <Sparkles className="h-6 w-6 text-primary" />
            <p className="text-xs text-muted-foreground">
              Спорт-консьерж Рондо.
              <br />
              Спросит погоду и проверит свободные корты.
            </p>
            <Link href="/agent">
              <Button size="sm" variant="outline" className="text-xs">
                Открыть Agent
              </Button>
            </Link>
          </div>
        </TabsContent>
      </Tabs>

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
