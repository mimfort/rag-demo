"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useParams } from "next/navigation";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { Plus, MessageSquare, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/lib/api";
import { useSettings } from "@/stores/settings";
import { cn } from "@/lib/utils";

export function ChatsList() {
  const router = useRouter();
  const params = useParams<{ chatId?: string }>();
  const activeChatId = params?.chatId ?? null;
  const queryClient = useQueryClient();
  const forgetSettings = useSettings((s) => s.forget);

  const chatsQuery = useQuery({
    queryKey: ["chats"],
    queryFn: api.listChats,
    refetchOnWindowFocus: false,
  });

  const createMutation = useMutation({
    mutationFn: () => api.createChat(null),
    onSuccess: (chat) => {
      queryClient.invalidateQueries({ queryKey: ["chats"] });
      router.push(`/chat/${chat.id}`);
    },
    onError: (err: Error) => toast.error("Не удалось создать чат: " + err.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteChat(id),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["chats"] });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      forgetSettings(data.chat_id);
      if (activeChatId === data.chat_id) {
        router.push("/");
      }
      toast.success("Чат удалён");
    },
    onError: (err: Error) => toast.error("Ошибка удаления: " + err.message),
  });

  const chats = chatsQuery.data ?? [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-border">
        <span className="text-xs font-semibold text-muted-foreground tracking-wider uppercase">
          Чаты
        </span>
        <Button
          size="icon"
          variant="ghost"
          className="h-7 w-7"
          onClick={() => createMutation.mutate()}
          disabled={createMutation.isPending}
          title="Новый чат"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      <Link
        href="/"
        className={cn(
          "flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:bg-accent transition-colors",
          activeChatId === null && "bg-accent text-foreground",
        )}
      >
        <MessageSquare className="h-3.5 w-3.5" />
        <span className="italic">без чата (stateless)</span>
      </Link>

      <ScrollArea className="flex-1">
        <div className="p-1 space-y-0.5">
          {chats.length === 0 && (
            <p className="text-xs text-muted-foreground px-3 py-4 text-center">
              Пока пусто. Нажми <Plus className="inline h-3 w-3" /> чтобы создать чат.
            </p>
          )}
          {chats.map((chat) => (
            <ChatRow
              key={chat.id}
              chatId={chat.id}
              title={chat.title}
              active={chat.id === activeChatId}
              onDelete={() => {
                if (confirm(`Удалить «${chat.title}»?`)) deleteMutation.mutate(chat.id);
              }}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

function ChatRow({
  chatId,
  title,
  active,
  onDelete,
}: {
  chatId: string;
  title: string;
  active: boolean;
  onDelete: () => void;
}) {
  return (
    <div className="group relative">
      <Link
        href={`/chat/${chatId}`}
        className={cn(
          "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors",
          "hover:bg-accent",
          active ? "bg-accent text-foreground" : "text-muted-foreground",
        )}
      >
        <MessageSquare className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate flex-1">{title}</span>
      </Link>
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onDelete();
        }}
        className="absolute right-1.5 top-1.5 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
        title="Удалить чат"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
