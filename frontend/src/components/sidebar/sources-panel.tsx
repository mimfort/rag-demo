"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { UploadCloud, FileText, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export function SourcesPanel() {
  const params = useParams<{ chatId?: string }>();
  const chatId = params?.chatId ?? null;
  const queryClient = useQueryClient();
  const fileRef = React.useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = React.useState(false);
  // Если есть активный чат — пользователь сам решает: грузить в общую базу
  // или в приватный пул чата.
  const [uploadToChat, setUploadToChat] = React.useState(true);

  const sourcesQuery = useQuery({
    queryKey: ["sources", chatId],
    queryFn: () => api.listSources(chatId),
    refetchOnWindowFocus: false,
  });

  const uploadMutation = useMutation({
    mutationFn: ({ file, targetChatId }: { file: File; targetChatId: string | null }) =>
      api.uploadFile(file, targetChatId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
      toast.success(`«${data.source}» — ${data.chunks} чанков`);
    },
    onError: (err: Error) => toast.error("Ошибка загрузки: " + err.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (source: string) => api.deleteSource(source),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
    onError: (err: Error) => toast.error("Ошибка удаления: " + err.message),
  });

  const handleFiles = React.useCallback(
    (files: FileList | null) => {
      if (!files || !files[0]) return;
      const targetChatId = chatId && uploadToChat ? chatId : null;
      uploadMutation.mutate({ file: files[0], targetChatId });
    },
    [chatId, uploadToChat, uploadMutation],
  );

  return (
    <div className="flex flex-col h-full border-t border-border">
      <div className="px-3 py-2 border-b border-border">
        <span className="text-xs font-semibold text-muted-foreground tracking-wider uppercase">
          Источники
        </span>
      </div>

      <div className="p-3">
        <label
          htmlFor="file-input"
          onDragEnter={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setDragOver(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer?.files);
          }}
          className={cn(
            "flex flex-col items-center justify-center gap-1 rounded-md border border-dashed border-border bg-card/50 p-4 text-center cursor-pointer transition-colors",
            dragOver && "border-primary bg-primary/5",
            uploadMutation.isPending && "opacity-50 pointer-events-none",
          )}
        >
          <input
            id="file-input"
            ref={fileRef}
            type="file"
            accept=".md,.markdown,.txt,.pdf"
            onChange={(e) => {
              handleFiles(e.target.files);
              if (fileRef.current) fileRef.current.value = "";
            }}
            className="hidden"
          />
          <UploadCloud className="h-5 w-5 text-muted-foreground" />
          <p className="text-xs text-muted-foreground">
            {uploadMutation.isPending ? "Загружаю…" : (
              <>
                Перетащи или <span className="text-primary font-semibold">выбери</span>
                <br />
                <span className="text-[10px]">.md / .txt / .pdf, до 5 МБ</span>
              </>
            )}
          </p>
        </label>

        {chatId && (
          <div className="flex items-center justify-between gap-2 mt-3 text-xs">
            <Label htmlFor="upload-to-chat" className="cursor-pointer">
              грузить в этот чат
            </Label>
            <Switch
              id="upload-to-chat"
              checked={uploadToChat}
              onCheckedChange={setUploadToChat}
            />
          </div>
        )}
      </div>

      <ScrollArea className="flex-1 px-2 pb-2">
        <div className="space-y-1">
          {sourcesQuery.data?.length === 0 && (
            <p className="text-[11px] text-muted-foreground px-2 py-4 text-center">
              Здесь пусто. Перетащи файл.
            </p>
          )}
          {sourcesQuery.data?.map((src) => (
            <SourceRow
              key={`${src.source}-${src.chat_id ?? "global"}`}
              name={src.source}
              chunks={src.chunks}
              isPrivate={src.chat_id !== null}
              onDelete={() => {
                if (confirm(`Удалить «${src.source}»?`)) deleteMutation.mutate(src.source);
              }}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

function SourceRow({
  name,
  chunks,
  isPrivate,
  onDelete,
}: {
  name: string;
  chunks: number;
  isPrivate: boolean;
  onDelete: () => void;
}) {
  return (
    <div className="group flex items-center gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-accent">
      <FileText className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
      <span className="flex-1 font-mono text-foreground truncate" title={name}>
        {name}
      </span>
      <span className={cn(
        "text-[10px] font-mono",
        isPrivate ? "text-violet-400" : "text-muted-foreground",
      )} title={isPrivate ? "Приватный для этого чата" : "Общая база"}>
        {chunks}ч.
      </span>
      <button
        type="button"
        onClick={onDelete}
        className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
        title="Удалить из базы"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
