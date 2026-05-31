"use client";

import * as React from "react";
import { ChevronDown, Sparkles, Database, MessageSquareOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import type { RagMode } from "@/lib/types";

interface Props {
  value: RagMode;
  onChange: (v: RagMode) => void;
}

const META: Record<
  RagMode,
  {
    label: string;
    Icon: React.ComponentType<{ className?: string }>;
    chipClass: string;
    hint: string;
  }
> = {
  auto: {
    label: "Auto",
    Icon: Sparkles,
    chipClass: "text-foreground",
    hint: "LLM решает: chitchat → без RAG, knowledge → с RAG",
  },
  on: {
    label: "On",
    Icon: Database,
    chipClass: "text-primary",
    hint: "Принудительно искать по базе на каждый вопрос",
  },
  off: {
    label: "Off",
    Icon: MessageSquareOff,
    chipClass: "text-muted-foreground",
    hint: "Обычный чат с LLM без обращения к базе знаний",
  },
};

export function RagModeChip({ value, onChange }: Props) {
  const [open, setOpen] = React.useState(false);
  // SSR совместимость: Zustand persist на сервере отдаёт default-значение,
  // а после регидрации клиент читает реальный rag_mode из localStorage.
  // Чтобы первый клиентский рендер совпал с серверным, показываем "auto"
  // до монтирования, потом переключаемся на реальное value.
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const effective: RagMode = mounted ? value : "auto";
  const cur = META[effective];
  const Cur = cur.Icon;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn("h-8 shrink-0 gap-1.5 px-2 font-medium", cur.chipClass)}
          title={`Режим RAG: ${cur.label} — ${cur.hint}`}
        >
          <Cur className="h-3.5 w-3.5" />
          <span className="text-xs">RAG: {cur.label}</span>
          <ChevronDown className="h-3 w-3 opacity-60" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-72 p-1.5">
        <p className="px-2 py-1 text-[11px] uppercase tracking-wider text-muted-foreground">
          Режим ответа
        </p>
        {(Object.keys(META) as RagMode[]).map((m) => {
          const Item = META[m].Icon;
          const selected = m === effective;
          return (
            <button
              key={m}
              type="button"
              onClick={() => {
                onChange(m);
                setOpen(false);
              }}
              className={cn(
                "w-full flex items-start gap-2 rounded-sm px-2 py-1.5 text-left",
                "hover:bg-accent transition-colors",
                selected && "bg-accent/60",
              )}
            >
              <Item className={cn("h-4 w-4 mt-0.5 shrink-0", META[m].chipClass)} />
              <div className="min-w-0">
                <div className="text-sm font-medium leading-tight">
                  {META[m].label}
                </div>
                <div className="text-[11px] text-muted-foreground leading-snug">
                  {META[m].hint}
                </div>
              </div>
            </button>
          );
        })}
      </PopoverContent>
    </Popover>
  );
}
