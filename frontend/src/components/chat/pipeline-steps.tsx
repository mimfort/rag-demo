"use client";

/**
 * Визуализация всех шагов retrieval-pipeline как «think aloud» AI:
 *   • роутер → знание / chitchat / meta / other
 *   • standalone-rewrite (если в чате с историей)
 *   • декомпозиция запроса на подвопросы
 *   • переформулировки (rewrite)
 *   • retrieve top-K + бейджи cos/bm25/rrf/rrk
 *   • контекст-расширение соседями
 *   • метрики времени по этапам
 *
 * Сворачивается в одну строку «5 шагов · 12.3 с», раскрывается в подробный
 * список — как у Perplexity / ChatGPT с «show steps».
 */

import * as React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import {
  ArrowRight,
  Brain,
  GitBranch,
  Layers,
  PencilLine,
  Route,
  Search,
  Sparkles,
} from "lucide-react";
import type { Chunk, Explain } from "@/lib/types";
import { cn, fmtMs } from "@/lib/utils";

interface Props {
  explain: Explain | null;
  chunks: Chunk[];
}

/** Один атомарный шаг pipeline. */
interface Step {
  key: string;
  icon: React.ReactNode;
  title: string;
  summary: string;
  details?: React.ReactNode;
  variant?: "default" | "success" | "warn" | "violet" | "primary";
}

function buildSteps(explain: Explain | null, chunks: Chunk[]): Step[] {
  if (!explain) return [];
  const steps: Step[] = [];

  // 1. Router
  if (explain.routed) {
    const intent = explain.route_intent ?? "knowledge";
    const variantMap: Record<string, Step["variant"]> = {
      knowledge: "success",
      chitchat: "primary",
      meta: "violet",
      other: "warn",
    };
    steps.push({
      key: "router",
      icon: <Route className="h-3.5 w-3.5" />,
      title: "Router",
      summary: `${intent} · ${fmtMs(explain.route_ms)}`,
      variant: variantMap[intent] ?? "default",
      details: (
        <div className="text-xs text-muted-foreground">
          {explain.route_reason}
          {explain.rag_skipped && (
            <div className="mt-1 text-amber-400">
              ↳ RAG пропущен, LLM ответил напрямую.
            </div>
          )}
          {explain.route_fallback && (
            <div className="mt-1 text-amber-400">↳ Сработал fallback на knowledge.</div>
          )}
        </div>
      ),
    });
    if (explain.rag_skipped) return steps; // дальше всё пропущено
  }

  // 2. Standalone rewrite (если в чате)
  if (explain.standalone_changed && explain.standalone_query) {
    steps.push({
      key: "standalone",
      icon: <Brain className="h-3.5 w-3.5" />,
      title: "History-aware rewrite",
      summary: `${explain.history_used} сообщений · ${fmtMs(explain.standalone_ms)}`,
      details: (
        <div className="text-xs">
          <div className="text-muted-foreground">переформулирован в:</div>
          <div className="mt-1 font-mono">{explain.standalone_query}</div>
        </div>
      ),
    });
  }

  // 3. Decompose
  if (explain.decomposed && explain.subqueries && explain.subqueries.length > 1) {
    steps.push({
      key: "decompose",
      icon: <GitBranch className="h-3.5 w-3.5" />,
      title: "Decompose",
      summary: `${explain.subqueries.length} подвопроса · ${fmtMs(explain.decompose_ms)}`,
      details: (
        <ul className="text-xs space-y-1">
          {explain.subqueries.map((sq, i) => (
            <li key={i} className="flex gap-2">
              <Badge variant="primary">q{i + 1}</Badge>
              <span>{sq}</span>
            </li>
          ))}
        </ul>
      ),
    });
  }

  // 4. Rewrite (formulations)
  if (explain.rewritten && explain.rewrite_groups && explain.rewrite_groups.length > 0) {
    const totalRewrites = explain.rewrite_groups.reduce(
      (sum, g) => sum + g.rewrites.length,
      0,
    );
    steps.push({
      key: "rewrite",
      icon: <PencilLine className="h-3.5 w-3.5" />,
      title: "Query rewrite",
      summary: `${totalRewrites} формулировок · ${fmtMs(explain.rewrite_ms)}`,
      details: (
        <div className="text-xs space-y-2">
          {explain.rewrite_groups.map((g, gi) => (
            <div key={gi}>
              {explain.rewrite_groups!.length > 1 && (
                <div className="text-muted-foreground mb-1">
                  Подзапрос {gi + 1}: {g.subquery}
                </div>
              )}
              <ul className="space-y-1">
                {g.rewrites.map((rw, ri) => (
                  <li key={ri} className="flex gap-2 ml-2">
                    <Badge variant="violet">q{gi + 1}.{ri + 1}</Badge>
                    <span>{rw}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      ),
    });
  }

  // 5. Retrieve (всегда показываем если есть chunks)
  if (chunks.length > 0) {
    steps.push({
      key: "retrieve",
      icon: <Search className="h-3.5 w-3.5" />,
      title: "Retrieve",
      summary:
        `${chunks.length} чанков · ${explain.search_mode}` +
        (explain.reranked ? ` + rerank (${fmtMs(explain.rerank_ms)})` : "") +
        (explain.mmr_applied ? ` + mmr λ=${explain.mmr_lambda?.toFixed(2)}` : ""),
      variant: "success",
      details: <ChunksList chunks={chunks} />,
    });
  }

  // 6. Context expansion
  if (explain.context_expanded && explain.neighbors_added) {
    steps.push({
      key: "expand",
      icon: <Layers className="h-3.5 w-3.5" />,
      title: "Expand context",
      summary: `+${explain.neighbors_added} соседей (радиус ±${explain.expand_radius})`,
      details: (
        <p className="text-xs text-muted-foreground">
          К финальным чанкам подмешаны соседи по chunk_index для построения
          связного контекста LLM.
        </p>
      ),
    });
  }

  // 7. Threshold filter — показываем только если что-то отрезали
  if (explain.filtered_by_threshold && explain.filtered_by_threshold > 0) {
    steps.push({
      key: "threshold",
      icon: <Sparkles className="h-3.5 w-3.5" />,
      title: "Threshold filter",
      summary: `отрезано ${explain.filtered_by_threshold} с rrk < ${explain.min_rerank_score.toFixed(2)}`,
      variant: "warn",
    });
  }

  return steps;
}

function ChunksList({ chunks }: { chunks: Chunk[] }) {
  return (
    <ul className="space-y-2">
      {chunks.map((c, i) => (
        <li
          key={`${c.source}-${c.chunk_index}-${i}`}
          id={`chunk-${i + 1}`}
          className="rounded-md border border-border bg-card/60 p-2 text-xs"
        >
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2">
              <Badge variant="primary">#{i + 1}</Badge>
              <span className="font-mono text-foreground">
                {c.source}#{c.chunk_index}
              </span>
              {c.original_rank != null && c.original_rank !== i + 1 && (
                <span className="font-mono text-[10px] text-muted-foreground">
                  #{c.original_rank} <ArrowRight className="inline h-2.5 w-2.5" /> #{i + 1}
                </span>
              )}
            </div>
            <div className="flex gap-1 flex-wrap justify-end">
              {c.vector_rank != null && (
                <Badge variant="primary" title="cosine similarity и ранг в vector-поиске">
                  cos {c.similarity.toFixed(3)}
                </Badge>
              )}
              {c.text_score != null && (
                <Badge variant="success" title="ts_rank_cd и ранг в text-поиске">
                  bm25 {c.text_score.toFixed(3)}
                </Badge>
              )}
              {c.rrf_score != null && (
                <Badge variant="warn" title="Reciprocal Rank Fusion">
                  rrf {c.rrf_score.toFixed(4)}
                </Badge>
              )}
              {c.reranker_score != null && (
                <Badge variant="violet" title="cross-encoder reranker">
                  rrk {c.reranker_score.toFixed(3)}
                </Badge>
              )}
            </div>
          </div>
          <p className="text-muted-foreground leading-relaxed line-clamp-3">{c.content}</p>
        </li>
      ))}
    </ul>
  );
}

export function PipelineSteps({ explain, chunks }: Props) {
  const steps = React.useMemo(() => buildSteps(explain, chunks), [explain, chunks]);
  if (!steps.length) return null;

  const totalMs =
    (explain?.route_ms ?? 0) +
    (explain?.standalone_ms ?? 0) +
    (explain?.decompose_ms ?? 0) +
    (explain?.rewrite_ms ?? 0) +
    (explain?.embed_ms ?? 0) +
    (explain?.rerank_ms ?? 0);

  return (
    <Accordion type="single" collapsible className="border-border">
      <AccordionItem value="steps" className="border-none">
        <AccordionTrigger className="py-1.5">
          <span className="flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5" />
            {steps.length} шагов · {fmtMs(totalMs)} · нажми чтобы раскрыть
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <ol className="space-y-2">
            {steps.map((s, i) => (
              <li key={s.key} className="flex gap-3">
                <div className="flex flex-col items-center pt-1">
                  <span
                    className={cn(
                      "flex h-6 w-6 items-center justify-center rounded-full border text-[10px] font-mono",
                      "bg-card",
                      s.variant === "success" && "border-emerald-500/40 text-emerald-400",
                      s.variant === "warn" && "border-amber-500/40 text-amber-400",
                      s.variant === "violet" && "border-violet-500/40 text-violet-400",
                      s.variant === "primary" && "border-primary/40 text-primary",
                      !s.variant && "border-border text-muted-foreground",
                    )}
                  >
                    {s.icon}
                  </span>
                  {i < steps.length - 1 && (
                    <div className="w-px flex-1 bg-border my-1" />
                  )}
                </div>
                <div className="flex-1 pb-2 min-w-0">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-semibold">{s.title}</span>
                    <span className="text-muted-foreground">{s.summary}</span>
                  </div>
                  {s.details && <div className="mt-1.5">{s.details}</div>}
                </div>
              </li>
            ))}
          </ol>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
