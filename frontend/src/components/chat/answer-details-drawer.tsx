"use client";

/**
 * Полный разбор retrieval-pipeline для конкретного ответа.
 * Открывается из MessageBubble по клику на иконку «детали».
 *
 * Секции:
 *   1. Pipeline steps — таймлайн как у Perplexity «show reasoning»
 *   2. Embedding запроса — модель/размерность/норма/время + превью вектора
 *   3. Cosine math — формула + подставленные числа для top-1 чанка
 *   4. Similarity distribution — гистограмма по ВСЕМ чанкам базы (с барами)
 *   5. User-prompt — что именно улетело в LLM (раскрывающийся блок)
 */

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { PipelineSteps } from "./pipeline-steps";
import type { Chunk, Explain } from "@/lib/types";
import { cn, fmtMs } from "@/lib/utils";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chunks: Chunk[];
  explain: Explain | null;
  prompt: string | null;
}

export function AnswerDetailsDrawer({
  open,
  onOpenChange,
  chunks,
  explain,
  prompt,
}: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent side="right" className="max-w-2xl flex flex-col gap-4">
        <DialogHeader>
          <DialogTitle>Как получен этот ответ</DialogTitle>
          <DialogDescription>
            Полный pipeline retrieval — шаги, числа, контекст для LLM
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 -mx-1 px-1 min-h-0">
          <div className="space-y-6 pr-2">
            {/* 1. Шаги pipeline */}
            <Section title="Шаги pipeline">
              <PipelineSteps explain={explain} chunks={chunks} />
            </Section>

            {/* 2. Embedding info */}
            <SectionEmbedding explain={explain} />

            {/* 3. Cosine math */}
            <SectionCosineMath explain={explain} />

            {/* 4. Distribution */}
            <SectionDistribution explain={explain} chunks={chunks} />

            {/* 5. Prompt */}
            <SectionPrompt prompt={prompt} />
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

function Section({
  title,
  children,
  hint,
}: {
  title: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <section className="space-y-2">
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {title}
        </h3>
        {hint && (
          <p className="text-[11px] text-muted-foreground/70 mt-0.5">{hint}</p>
        )}
      </div>
      <div>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------- */
/* 2. Embedding запроса                                           */
/* ------------------------------------------------------------- */
function SectionEmbedding({ explain }: { explain: Explain | null }) {
  if (!explain || !explain.query_preview?.length) return null;
  const preview = explain.query_preview;
  const isNormalized = Math.abs((explain.query_norm ?? 0) - 1.0) < 0.01;

  return (
    <Section
      title="Эмбеддинг запроса"
      hint="как запрос превратился в вектор"
    >
      <div className="rounded-md border border-border bg-card p-3 space-y-3 text-xs">
        <div className="grid grid-cols-2 gap-y-1 gap-x-4">
          <KV label="модель" value={explain.embed_model} mono />
          <KV
            label="размерность"
            value={String(explain.embed_dim)}
            mono
          />
          <KV
            label="‖q‖ (норма)"
            value={(explain.query_norm ?? 0).toFixed(4)}
            mono
          />
          <KV label="время" value={fmtMs(explain.embed_ms)} mono />
        </div>

        <div>
          <div className="text-muted-foreground mb-1">
            Превью первых {preview.length} компонент:
          </div>
          <div className="font-mono text-[11px] text-muted-foreground bg-secondary/40 rounded p-2 break-all leading-relaxed max-h-24 overflow-auto">
            [{preview.map((v) => v.toFixed(4)).join(", ")}, …]
          </div>
        </div>

        {isNormalized && (
          <p className="text-[11px] text-muted-foreground italic">
            Норма ≈ 1 — bge-m3 возвращает уже нормализованные векторы. Для
            нормализованных cosine ≡ скалярному произведению.
          </p>
        )}
      </div>
    </Section>
  );
}

function KV({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-muted-foreground">{label}:</span>
      <span className={cn(mono && "font-mono", "text-foreground")}>{value}</span>
    </div>
  );
}

/* ------------------------------------------------------------- */
/* 3. Cosine similarity — пошаговый расчёт                        */
/* ------------------------------------------------------------- */
function SectionCosineMath({ explain }: { explain: Explain | null }) {
  if (
    !explain ||
    explain.top_similarity == null ||
    explain.top_chunk_norm == null ||
    explain.top_dot == null
  ) {
    return null;
  }
  const dot = explain.top_dot;
  const qn = explain.query_norm;
  const dn = explain.top_chunk_norm;
  const sim = explain.top_similarity;

  return (
    <Section
      title="Cosine similarity"
      hint="как Postgres посчитал близость top-1 чанка к запросу"
    >
      <div className="rounded-md border border-border bg-card p-3 space-y-3 font-mono text-xs">
        <div className="text-muted-foreground italic leading-relaxed">
          {"            q · d"}
          <br />
          {"sim(q,d) = ─────────"}
          <br />
          {"          ‖q‖ · ‖d‖"}
        </div>
        <div className="border-t border-border pt-3 space-y-1">
          <div>
            q · d = <span className="text-foreground">{dot.toFixed(6)}</span>
          </div>
          <div>
            ‖q‖ = <span className="text-foreground">{(qn ?? 0).toFixed(6)}</span>
          </div>
          <div>
            ‖d‖ = <span className="text-foreground">{dn.toFixed(6)}</span>
          </div>
          <div className="text-muted-foreground">─────────────────────</div>
          <div>
            sim ={" "}
            <span className="text-foreground">{dot.toFixed(6)}</span> / (
            <span className="text-foreground">{(qn ?? 0).toFixed(6)}</span> ×{" "}
            <span className="text-foreground">{dn.toFixed(6)}</span>) ={" "}
            <span className="text-primary font-semibold">{sim.toFixed(6)}</span>
          </div>
        </div>
        <p className="text-[11px] text-muted-foreground italic font-sans">
          Postgres через оператор <code className="font-mono">1 - (embedding &lt;=&gt; query::vector)</code> вернул то же число.
        </p>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------- */
/* 4. Distribution по всей базе                                   */
/* ------------------------------------------------------------- */
function SectionDistribution({
  explain,
  chunks,
}: {
  explain: Explain | null;
  chunks: Chunk[];
}) {
  if (!explain?.all_scores?.length) return null;
  const all = explain.all_scores;
  const topKeys = new Set(
    chunks.map((c) => `${c.source}#${c.chunk_index}`),
  );

  return (
    <Section
      title={`Similarity по всей базе (${all.length})`}
      hint="видно разрыв между релевантными и нерелевантными чанками"
    >
      <Accordion type="single" collapsible>
        <AccordionItem value="dist" className="border-none">
          <AccordionTrigger>
            показать распределение
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-1 mt-2">
              {all.map((s) => {
                const isTop = topKeys.has(`${s.source}#${s.chunk_index}`);
                const pct = Math.max(0, Math.min(100, s.similarity * 100));
                return (
                  <div
                    key={`${s.source}#${s.chunk_index}`}
                    className="grid grid-cols-[1fr_3rem_2fr] gap-2 items-center text-[11px] font-mono"
                  >
                    <span
                      className={cn(
                        "truncate",
                        isTop ? "text-foreground" : "text-muted-foreground",
                      )}
                      title={`${s.source}#${s.chunk_index}`}
                    >
                      {s.source}#{s.chunk_index}
                    </span>
                    <span
                      className={cn(
                        "text-right",
                        isTop ? "text-primary font-semibold" : "text-foreground",
                      )}
                    >
                      {s.similarity.toFixed(3)}
                    </span>
                    <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-full",
                          isTop ? "bg-primary" : "bg-muted-foreground/50",
                        )}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Section>
  );
}

/* ------------------------------------------------------------- */
/* 5. Prompt                                                       */
/* ------------------------------------------------------------- */
function SectionPrompt({ prompt }: { prompt: string | null }) {
  if (!prompt) return null;
  return (
    <Section
      title="User-prompt"
      hint="что улетело в LLM вместе с контекстом"
    >
      <Accordion type="single" collapsible>
        <AccordionItem value="prompt" className="border-none">
          <AccordionTrigger>раскрыть промпт</AccordionTrigger>
          <AccordionContent>
            <pre className="text-[11px] font-mono whitespace-pre-wrap bg-card border border-border rounded p-3 max-h-96 overflow-auto leading-relaxed">
              {prompt}
            </pre>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Section>
  );
}
