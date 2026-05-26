"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DEFAULT_SETTINGS, type RetrievalSettings, type SearchMode } from "@/lib/types";
import { PRESETS, useSettings } from "@/stores/settings";
import { cn } from "@/lib/utils";
import { Zap, Scale, Sparkles } from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  chatId: string | null;
}

/**
 * Главный drawer настроек retrieval для конкретного чата (или default).
 * Изменения сохраняются мгновенно в zustand store (он persist'ит в localStorage).
 */
export function SettingsDrawer({ open, onOpenChange, chatId }: Props) {
  const settings = useSettings((s) => s.getFor(chatId));
  const set = useSettings((s) => s.setFor);
  const replace = useSettings((s) => s.replaceFor);

  const update = React.useCallback(
    (patch: Partial<RetrievalSettings>) => set(chatId, patch),
    [chatId, set],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent side="right">
        <DialogHeader>
          <DialogTitle>Настройки retrieval</DialogTitle>
          <DialogDescription>
            {chatId ? "Применяются только к этому чату" : "Default — для режима без чата"}
          </DialogDescription>
        </DialogHeader>

        {/* Пресеты */}
        <div className="grid grid-cols-3 gap-2">
          <PresetButton
            icon={<Zap className="h-3.5 w-3.5" />}
            label="Быстрый"
            onClick={() =>
              replace(chatId, { ...DEFAULT_SETTINGS, ...PRESETS.fast })
            }
          />
          <PresetButton
            icon={<Scale className="h-3.5 w-3.5" />}
            label="Сбалансированный"
            onClick={() =>
              replace(chatId, { ...DEFAULT_SETTINGS, ...PRESETS.balanced })
            }
          />
          <PresetButton
            icon={<Sparkles className="h-3.5 w-3.5" />}
            label="Максимум"
            onClick={() =>
              replace(chatId, { ...DEFAULT_SETTINGS, ...PRESETS.thorough })
            }
          />
        </div>

        <ScrollArea className="flex-1 -mx-1 px-1">
          <div className="space-y-5 py-2">
            {/* Generation */}
            <Section title="Генерация">
              <Row
                label="streaming"
                hint="постепенная подача токенов через SSE"
              >
                <Switch
                  checked={settings.streaming}
                  onCheckedChange={(v) => update({ streaming: v })}
                />
              </Row>
              <Row
                label="auto-route"
                hint="LLM решает нужен ли RAG (chitchat / meta / other → без поиска)"
              >
                <Switch
                  checked={settings.auto_route}
                  onCheckedChange={(v) => update({ auto_route: v })}
                />
              </Row>
            </Section>

            {/* Retrieval */}
            <Section title="Retrieval">
              <Row label="top_k" hint="сколько чанков подаём в LLM">
                <NumberStepper
                  value={settings.top_k}
                  min={1}
                  max={20}
                  onChange={(v) => update({ top_k: v })}
                />
              </Row>
              <Row label="search mode" hint="vector / text / hybrid">
                <ModeToggle
                  value={settings.search_mode}
                  onChange={(v) => update({ search_mode: v })}
                />
              </Row>
            </Section>

            {/* Query understanding */}
            <Section title="Query understanding">
              <Row label="decompose" hint="разбить составной вопрос на подзапросы">
                <Switch
                  checked={settings.decompose}
                  onCheckedChange={(v) => update({ decompose: v })}
                />
              </Row>
              <Row label="query rewrite" hint="N разных формулировок одного вопроса">
                <Switch
                  checked={settings.rewrite}
                  onCheckedChange={(v) => update({ rewrite: v })}
                />
              </Row>
              {settings.rewrite && (
                <Row label="N формулировок" hint="">
                  <NumberStepper
                    value={settings.rewrite_n}
                    min={1}
                    max={5}
                    onChange={(v) => update({ rewrite_n: v })}
                  />
                </Row>
              )}
            </Section>

            {/* Reranking */}
            <Section title="Reranking">
              <Row label="rerank (cross-encoder)" hint="bge-reranker-v2-m3 поверх top-15">
                <Switch
                  checked={settings.rerank}
                  onCheckedChange={(v) => update({ rerank: v })}
                />
              </Row>
              <Row
                label="per-subquery rerank"
                hint="rerank на каждом подзапросе отдельно (нужен decompose+rerank)"
              >
                <Switch
                  checked={settings.rerank_per_subquery}
                  onCheckedChange={(v) => {
                    if (v) update({ rerank_per_subquery: true, rerank: true, decompose: true });
                    else update({ rerank_per_subquery: false });
                  }}
                />
              </Row>
              <Row
                label="min rerank score"
                hint="отрезать чанки с rrk ниже порога"
                badge={settings.min_rerank_score.toFixed(2)}
              >
                <Slider
                  value={settings.min_rerank_score}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(v) => update({ min_rerank_score: v })}
                />
              </Row>
            </Section>

            {/* MMR */}
            <Section title="Diversification (MMR)">
              <Row label="mmr" hint="балансирует релевантность и разнообразие">
                <Switch
                  checked={settings.mmr}
                  onCheckedChange={(v) => update({ mmr: v })}
                />
              </Row>
              {settings.mmr && (
                <Row
                  label="λ"
                  hint="1.0 = чистая релевантность · 0.0 = чистое разнообразие"
                  badge={settings.mmr_lambda.toFixed(2)}
                >
                  <Slider
                    value={settings.mmr_lambda}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={(v) => update({ mmr_lambda: v })}
                  />
                </Row>
              )}
            </Section>

            {/* Context expansion */}
            <Section title="Context expansion">
              <Row
                label="expand context"
                hint="добавить соседей чанков в промпт LLM"
              >
                <Switch
                  checked={settings.expand_context}
                  onCheckedChange={(v) => update({ expand_context: v })}
                />
              </Row>
              {settings.expand_context && (
                <Row label="радиус ±" hint="" badge={String(settings.expand_radius)}>
                  <NumberStepper
                    value={settings.expand_radius}
                    min={1}
                    max={3}
                    onChange={(v) => update({ expand_radius: v })}
                  />
                </Row>
              )}
            </Section>

            {/* Threshold */}
            <Section title="Cosine threshold">
              <Row
                label="min similarity"
                hint="если top-1 cos ниже — плашка-предупреждение"
                badge={settings.min_similarity.toFixed(2)}
              >
                <Slider
                  value={settings.min_similarity}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(v) => update({ min_similarity: v })}
                />
              </Row>
            </Section>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

/* ----- partials ----- */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">
        {title}
      </h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Row({
  label,
  hint,
  badge,
  children,
}: {
  label: string;
  hint?: string;
  badge?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-3">
        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2">
            <Label className="text-sm text-foreground">{label}</Label>
            {badge && (
              <Badge variant="outline" className="font-mono">
                {badge}
              </Badge>
            )}
          </div>
          {hint && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">{hint}</p>
          )}
        </div>
        <div className="shrink-0">{children}</div>
      </div>
    </div>
  );
}

function PresetButton({
  icon,
  label,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <Button variant="outline" size="sm" onClick={onClick} className="text-xs gap-1">
      {icon}
      {label}
    </Button>
  );
}

function NumberStepper({
  value,
  min,
  max,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-1">
      <Button
        variant="outline"
        size="icon"
        className="h-7 w-7"
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
      >
        −
      </Button>
      <Input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const v = parseInt(e.target.value || "0", 10);
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
        className="h-7 w-14 text-center font-mono"
      />
      <Button
        variant="outline"
        size="icon"
        className="h-7 w-7"
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
      >
        +
      </Button>
    </div>
  );
}

function ModeToggle({
  value,
  onChange,
}: {
  value: SearchMode;
  onChange: (v: SearchMode) => void;
}) {
  const modes: SearchMode[] = ["vector", "text", "hybrid"];
  return (
    <div className="flex rounded-md border border-border overflow-hidden">
      {modes.map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          className={cn(
            "px-2.5 py-1 text-[11px] font-mono transition-colors",
            value === m
              ? "bg-primary text-primary-foreground"
              : "bg-card text-muted-foreground hover:bg-accent",
          )}
        >
          {m}
        </button>
      ))}
    </div>
  );
}
