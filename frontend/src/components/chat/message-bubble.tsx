"use client";

import * as React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Chunk, Explain } from "@/lib/types";
import { AnswerDetailsDrawer } from "./answer-details-drawer";

interface Props {
  role: "user" | "assistant";
  content: string;
  /** Только для assistant — данные о retrieval-пайплайне для drawer'а. */
  explain?: Explain | null;
  chunks?: Chunk[];
  prompt?: string | null;
  /** Идёт ли стриминг прямо сейчас (показываем «думает…»). */
  streaming?: boolean;
}

/**
 * Парсит "...текст [1] ещё [2]..." и заменяет [N] на кликабельные чипы,
 * которые скроллят к соответствующему чанку (id="chunk-N") и подсвечивают его.
 */
function renderWithCitations(text: string, chunksCount: number) {
  const parts = text.split(/\[(\d+)\]/g);
  return parts.map((piece, i) => {
    if (i % 2 === 0) return piece;
    const n = parseInt(piece, 10);
    const alive = n >= 1 && n <= chunksCount;
    return (
      <button
        key={`cite-${i}`}
        type="button"
        className={cn("citation-chip", !alive && "dead")}
        onClick={(e) => {
          e.preventDefault();
          if (!alive) return;
          const el = document.getElementById(`chunk-${n}`);
          if (!el) return;
          el.scrollIntoView({ behavior: "smooth", block: "center" });
          el.classList.remove("chunk-flash");
          // force reflow для повторного старта анимации
          void (el as HTMLElement).offsetWidth;
          el.classList.add("chunk-flash");
        }}
        title={alive ? `Перейти к фрагменту #${n}` : `Цитата на несуществующий фрагмент`}
      >
        [{n}]
      </button>
    );
  });
}

/**
 * AssistantContent — кастомный рендер markdown с заменой [N] на чипы.
 * react-markdown работает с компонентами; нам надо перехватить
 * текстовые ноды в параграфах.
 */
function AssistantContent({ text, chunksCount }: { text: string; chunksCount: number }) {
  // Простое решение: преобразуем весь текст один раз через split + map в массив React-нод.
  // markdown отрендерим как обычный текст, но через ReactMarkdown — чтобы поддержать списки/код.
  // Хитрость: подменяем компонент `p` чтобы пройтись по children и заменить [N].
  const transformChildren = (children: React.ReactNode): React.ReactNode => {
    return React.Children.map(children, (child) => {
      if (typeof child === "string") {
        return renderWithCitations(child, chunksCount);
      }
      if (React.isValidElement(child)) {
        const props = child.props as { children?: React.ReactNode };
        if (props.children !== undefined) {
          return React.cloneElement(
            child as React.ReactElement<{ children?: React.ReactNode }>,
            undefined,
            transformChildren(props.children),
          );
        }
      }
      return child;
    });
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children, ...props }) => <p {...props}>{transformChildren(children)}</p>,
        li: ({ children, ...props }) => <li {...props}>{transformChildren(children)}</li>,
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

export function MessageBubble({
  role,
  content,
  explain,
  chunks = [],
  prompt = null,
  streaming,
}: Props) {
  const [detailsOpen, setDetailsOpen] = React.useState(false);
  // Условие «ручной bypass»: RAG был пропущен, но не из-за router'а
  // (router заполнил бы routed=true и route_intent). Если routed=false и
  // rag_skipped=true — значит пользователь явно выбрал режим Off.
  const isManualBypass =
    role === "assistant" &&
    explain != null &&
    explain.rag_skipped === true &&
    !explain.routed;
  const hasDetails =
    role === "assistant" && explain != null && !isManualBypass;

  return (
    <div className={cn("flex gap-3 group", role === "user" ? "flex-row-reverse" : "flex-row", isManualBypass && "mb-5")}>
      <div
        className={cn(
          "h-7 w-7 shrink-0 rounded-full flex items-center justify-center border",
          role === "user"
            ? "bg-primary/10 border-primary/40 text-primary"
            : "bg-secondary border-border text-foreground",
        )}
      >
        {role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
      </div>

      <div className={cn(
        "max-w-[calc(100%-3rem)] flex-1 min-w-0",
        role === "user" ? "items-end" : "items-start",
      )}>
        <div className="relative">
          <div
            className={cn(
              "rounded-2xl px-4 py-3",
              role === "user"
                ? "bg-primary/10 border border-primary/30"
                : "bg-card border border-border",
            )}
          >
            {role === "user" ? (
              <p className="prose-rag whitespace-pre-wrap">{content}</p>
            ) : (
              <div className="prose-rag">
                {content ? (
                  <>
                    <AssistantContent text={content} chunksCount={chunks.length} />
                    {/* Мигающий курсор пока идёт стрим — визуальный признак */}
                    {streaming && (
                      <span
                        aria-hidden
                        className="inline-block w-[6px] h-[1em] -mb-[2px] ml-0.5 bg-primary animate-pulse align-middle"
                      />
                    )}
                  </>
                ) : streaming ? (
                  <span className="text-muted-foreground italic">
                    Думаю
                    <span className="inline-flex gap-0.5 ml-1">
                      <span className="animate-pulse [animation-delay:0ms]">·</span>
                      <span className="animate-pulse [animation-delay:200ms]">·</span>
                      <span className="animate-pulse [animation-delay:400ms]">·</span>
                    </span>
                  </span>
                ) : (
                  <span className="text-muted-foreground italic">Пусто</span>
                )}
              </div>
            )}
          </div>

          {/* Иконка деталей: появляется как только есть explain (даже во время
              стрима — drawer обновляется в реальном времени). Во время
              стрима кнопка видна всегда (опасения «нечего показать»
              отпадают — meta приходит до токенов). После завершения —
              на hover. */}
          {hasDetails && (
            <button
              type="button"
              onClick={() => setDetailsOpen(true)}
              className={cn(
                "absolute -right-2 -bottom-2 h-7 w-7 rounded-full",
                "flex items-center justify-center",
                "bg-card border text-muted-foreground",
                "transition-all shadow-md",
                "hover:text-primary hover:border-primary",
                streaming
                  ? "opacity-100 border-primary/60 text-primary animate-pulse"
                  : "opacity-0 group-hover:opacity-100 border-border",
              )}
              title={
                streaming
                  ? "Идёт обработка — можно посмотреть шаги в реальном времени"
                  : "Как я дошёл до этого ответа"
              }
            >
              <Sparkles className="h-3.5 w-3.5" />
            </button>
          )}
          {isManualBypass && (
            <div className="absolute -bottom-5 left-0 text-[10px] text-muted-foreground">
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-card px-2 py-0.5">
                <span aria-hidden>○</span>
                RAG не использовался — режим обычного чата
              </span>
            </div>
          )}
        </div>
      </div>

      {hasDetails && (
        <AnswerDetailsDrawer
          open={detailsOpen}
          onOpenChange={setDetailsOpen}
          chunks={chunks}
          explain={explain!}
          prompt={prompt}
        />
      )}
    </div>
  );
}
