"use client";

import * as React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Send, StopCircle, Bot, User, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { agentApi } from "@/lib/agent-api";
import type { AgentMessage, TraceEvent, ClarifyData } from "@/lib/agent-types";
import { useAgentUi } from "@/stores/agent-ui";
import { TraceTimeline } from "./trace-timeline";
import { ClarifyPrompt } from "./clarify-prompt";

export function AgentChat() {
  const queryClient = useQueryClient();
  const draft = useAgentUi((s) => s.draft);
  const setDraft = useAgentUi((s) => s.setDraft);
  const appendTrace = useAgentUi((s) => s.appendTrace);
  const setAnswer = useAgentUi((s) => s.setAnswer);
  const setFinished = useAgentUi((s) => s.setFinished);
  const setPendingClarify = useAgentUi((s) => s.setPendingClarify);

  const [text, setText] = React.useState("");
  const bottomRef = React.useRef<HTMLDivElement>(null);

  const messagesQuery = useQuery({
    queryKey: ["agent", "messages"],
    queryFn: () => agentApi.messages(),
  });

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messagesQuery.data?.length, draft?.trace.length]);

  const openStream = React.useCallback(
    (url: string) => {
      const es = new EventSource(url);
      // Флаг штатного закрытия: clarify-пауза и done/error закрывают сокет
      // намеренно, и браузер всё равно выстрелит native error-событием (без
      // .data). Без этого флага на каждую паузу показывался бы ложный тост.
      let closedClean = false;
      const closeClean = () => {
        closedClean = true;
        es.close();
      };
      const abort = () => closeClean();
      // Обновляем только abort, остальной draft не трогаем.
      useAgentUi.setState((s) => (s.draft ? { draft: { ...s.draft, abort } } : {}));

      const onEvent = (type: TraceEvent["type"]) => (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          const ev: TraceEvent = { type, timestamp: new Date().toISOString(), data };
          appendTrace(ev);

          if (type === "clarify") {
            setPendingClarify(data as ClarifyData);
            closeClean(); // граф на паузе; продолжим через resume
            return;
          }
          if (type === "final_answer" && typeof data.text === "string") {
            setAnswer(data.text);
          }
          if (type === "done" || type === "error") {
            setFinished(true);
            closeClean();
            queryClient.invalidateQueries({ queryKey: ["agent", "messages"] });
            setTimeout(() => setDraft(null), 800);
          }
        } catch (err) {
          console.error("SSE parse failed", err);
        }
      };

      es.addEventListener("clarify", onEvent("clarify"));
      es.addEventListener("node_start", onEvent("node_start"));
      es.addEventListener("tool_call", onEvent("tool_call"));
      es.addEventListener("tool_result", onEvent("tool_result"));
      es.addEventListener("verify", onEvent("verify"));
      es.addEventListener("final_answer", onEvent("final_answer"));
      es.addEventListener("done", onEvent("done"));
      es.addEventListener("error", (e) => {
        const msgEv = e as MessageEvent;
        if (msgEv.data) {
          onEvent("error")(msgEv);
        } else if (!closedClean) {
          // native error без .data на НЕштатном разрыве — настоящий обрыв связи.
          toast.error("SSE-соединение прервано");
          setFinished(true);
          es.close();
          setDraft(null);
        }
      });
    },
    [appendTrace, setAnswer, setFinished, setDraft, setPendingClarify, queryClient],
  );

  const send = React.useCallback(() => {
    const q = text.trim();
    if (!q || draft) return;
    setText("");
    setDraft({
      userQuery: q,
      trace: [],
      answer: "",
      finished: false,
      abort: null,
      pendingClarify: null,
    });
    openStream(agentApi.buildStreamUrl(q));
  }, [text, draft, setDraft, openStream]);

  const onConfirm = React.useCallback(() => {
    const pc = draft?.pendingClarify;
    if (!pc) return;
    setPendingClarify(null);
    openStream(agentApi.buildResumeUrl({ threadId: pc.thread_id, confirmed: true }));
  }, [draft, setPendingClarify, openStream]);

  const onReject = React.useCallback(
    (correction: string) => {
      const pc = draft?.pendingClarify;
      if (!pc) return;
      setPendingClarify(null);
      openStream(
        agentApi.buildResumeUrl({ threadId: pc.thread_id, confirmed: false, correction }),
      );
    },
    [draft, setPendingClarify, openStream],
  );

  const stop = React.useCallback(() => {
    draft?.abort?.();
    setDraft(null);
  }, [draft, setDraft]);

  const messages: AgentMessage[] = messagesQuery.data ?? [];
  // Во время ожидания подтверждения не показываем «стоп»/курсор стрима.
  const isStreaming = draft !== null && !draft.finished && !draft.pendingClarify;

  return (
    <div className="flex h-full flex-col">
      <div className="px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold">Спорт-консьерж Рондо</h2>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Спрашивай про погоду и свободные корты — агент сам подёргает API.
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
          {messages.length === 0 && !draft && (
            <div className="flex flex-col items-center text-center py-16 gap-2">
              <Sparkles className="h-8 w-8 text-primary" />
              <p className="text-sm text-muted-foreground max-w-md">
                Спроси, например: <br />
                «Найди солнечный день на следующей неделе со свободным кортом».
              </p>
            </div>
          )}

          {messages.map((m) => (
            <Bubble
              key={m.id}
              role={m.role}
              content={m.content}
              trace={m.trace}
            />
          ))}

          {draft && (
            <>
              <Bubble role="user" content={draft.userQuery} />
              <Bubble
                role="assistant"
                content={draft.answer || (draft.finished ? "(пусто)" : "Думаю…")}
                trace={draft.trace}
                streaming={isStreaming}
              />
              {draft.pendingClarify && (
                <div className="ml-10">
                  <ClarifyPrompt
                    data={draft.pendingClarify}
                    onConfirm={onConfirm}
                    onReject={onReject}
                  />
                </div>
              )}
            </>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="border-t border-border bg-background p-4">
        <div className="mx-auto max-w-3xl flex items-end gap-2 rounded-2xl border border-border bg-card p-2">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Спроси что-нибудь… (Enter — отправить)"
            className="flex-1 min-h-[40px] max-h-[200px] resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 px-2 py-2"
            rows={1}
          />
          {isStreaming ? (
            <Button variant="destructive" size="icon" onClick={stop} title="Прервать" className="h-9 w-9 shrink-0">
              <StopCircle className="h-4 w-4" />
            </Button>
          ) : (
            <Button size="icon" onClick={send} disabled={!text.trim()} title="Отправить" className="h-9 w-9 shrink-0">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function Bubble({
  role,
  content,
  trace,
  streaming,
}: {
  role: "user" | "assistant";
  content: string;
  trace?: TraceEvent[] | null;
  streaming?: boolean;
}) {
  return (
    <div className={cn("flex gap-3", role === "user" ? "flex-row-reverse" : "flex-row")}>
      <div className={cn(
        "h-7 w-7 shrink-0 rounded-full flex items-center justify-center border",
        role === "user"
          ? "bg-primary/10 border-primary/40 text-primary"
          : "bg-secondary border-border text-foreground",
      )}>
        {role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
      </div>
      <div className="max-w-[calc(100%-3rem)] flex-1 min-w-0">
        <div className={cn(
          "rounded-2xl px-4 py-3",
          role === "user" ? "bg-primary/10 border border-primary/30" : "bg-card border border-border",
        )}>
          <p className="prose-rag whitespace-pre-wrap">{content}</p>
          {streaming && <span className="ml-1 inline-block w-[6px] h-[1em] bg-primary animate-pulse align-middle" />}
          {role === "assistant" && trace && trace.length > 0 && (
            <div className="mt-2 border-t border-border pt-2">
              <TraceTimeline trace={trace} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
