"use client";

import * as React from "react";
import { ChevronRight, Wrench, MessageSquare, Brain, CheckCircle2, AlertCircle, HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TraceEvent } from "@/lib/agent-types";

interface Props {
  event: TraceEvent;
}

function eventIcon(type: TraceEvent["type"]) {
  switch (type) {
    case "clarify":      return HelpCircle;
    case "node_start":   return Brain;
    case "tool_call":    return Wrench;
    case "tool_result":  return CheckCircle2;
    case "final_answer": return MessageSquare;
    case "done":         return CheckCircle2;
    case "error":        return AlertCircle;
  }
}

function eventTitle(ev: TraceEvent): string {
  switch (ev.type) {
    case "clarify": {
      const interp = ev.data.interpretation as string;
      const round = (ev.data.round as number) ?? 0;
      return `🤔 Уточнение${round > 1 ? ` (круг ${round})` : ""}: «${interp}»`;
    }
    case "node_start": {
      const node = ev.data.node as string;
      if (node === "agent") return "Думаю...";
      if (node === "interpret") return "Интерпретирую запрос…";
      if (node === "confirm") return "Уточнение";
      return `Узел: ${node}`;
    }
    case "tool_call": {
      const name = ev.data.name as string;
      const args = ev.data.args as Record<string, unknown>;
      const argStr = Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
      return `🔧 ${name}(${argStr})`;
    }
    case "tool_result": {
      const name = ev.data.name as string;
      const r = ev.data.result as Record<string, unknown> | string;
      if (typeof r === "object" && r !== null) {
        if ("error" in r) return `${name} → ошибка: ${r.error}`;
        if ("summary" in r) return `${name} → ${r.summary}`;
      }
      return `${name} → готово`;
    }
    case "final_answer": return "Финальный ответ";
    case "done":         return `Готово (итераций: ${ev.data.iterations})`;
    case "error":        return `Ошибка: ${ev.data.message}`;
  }
}

function eventColor(type: TraceEvent["type"]): string {
  switch (type) {
    case "error":         return "text-red-400";
    case "final_answer":  return "text-primary";
    case "tool_call":     return "text-amber-400";
    case "tool_result":   return "text-emerald-400";
    case "clarify":       return "text-amber-400";
    default:              return "text-muted-foreground";
  }
}

export function TraceStep({ event }: Props) {
  const [open, setOpen] = React.useState(false);
  const Icon = eventIcon(event.type);
  const title = eventTitle(event);
  const color = eventColor(event.type);

  return (
    <div className="flex flex-col text-[11px]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "flex items-start gap-1.5 text-left hover:bg-accent/40 rounded px-1 py-0.5 transition-colors",
          color,
        )}
      >
        <ChevronRight className={cn("h-3 w-3 mt-0.5 shrink-0 transition-transform", open && "rotate-90")} />
        <Icon className="h-3 w-3 mt-0.5 shrink-0" />
        <span className="break-words">{title}</span>
      </button>
      {open && (
        <pre className="ml-5 mt-1 p-2 rounded bg-card/60 border border-border overflow-x-auto text-[10px] leading-tight">
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  );
}
