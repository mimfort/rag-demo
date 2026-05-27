# RAG mode chip-toggle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-state chip-toggle (Auto / On / Off) to the composer
that controls whether RAG is used for the next user message.

**Architecture:** Frontend хранит `rag_mode` per-chat в Zustand-persist.
В API-вызов мапим `rag_mode` → существующий `auto_route` + новый
`bypass_rag` флаг. На бэке `bypass_rag` пропускает retrieve и идёт
прямой ответ через LLM с new general system-prompt. UI меняет
рендеринг сообщения, если RAG был bypass'нут вручную.

**Tech Stack:** Next.js 15 + React 19 + TypeScript + Zustand + TanStack
Query + shadcn/ui (radix-popover, ещё не как UI-компонент — добавим),
FastAPI + Pydantic + httpx.

**Spec:** `docs/superpowers/specs/2026-05-27-rag-mode-toggle-design.md`

---

## File map

**Backend:**
- Modify: `web/server.py` — `AskRequest`, `_generate_direct_answer`, новый `_GENERAL_SYSTEM`, ветки в `ask` и `ask_stream`.

**Frontend (types + state):**
- Modify: `frontend/src/lib/types.ts` — `RagMode`, поле, дефолт.
- Modify: `frontend/src/stores/settings.ts` — миграция, обновлённые пресеты.
- Modify: `frontend/src/lib/api.ts` — маппинг в `ask()` и `buildStreamUrl()`.

**Frontend (UI primitives):**
- Create: `frontend/src/components/ui/popover.tsx` — shadcn-обёртка над radix-popover.

**Frontend (chat UI):**
- Create: `frontend/src/components/chat/rag-mode-chip.tsx`.
- Modify: `frontend/src/components/chat/composer.tsx` — встроить chip.
- Modify: `frontend/src/components/chat/chat-view.tsx` — прокинуть проп.
- Modify: `frontend/src/components/chat/message-bubble.tsx` — бейдж «RAG не использовался».
- Modify: `frontend/src/components/settings/settings-drawer.tsx` — убрать auto-route switch.

---

## Pre-flight

- [ ] **Step 0: Убедиться, что dev-окружение запускается**

Run:
```bash
docker compose up -d
uvicorn web.server:app --reload --host 0.0.0.0 --port 8000 &
( cd frontend && npm install && npm run dev ) &
```

Открыть http://localhost:3000 и `curl http://localhost:8000/api/health`.
Ожидание: оба отвечают, `chunks_in_db > 0` (если нет — запустить
`python ingest.py`).

---

### Task 1: Backend — добавить `bypass_rag` и общий system-prompt

**Files:**
- Modify: `web/server.py`

- [ ] **Step 1: Добавить новый system-prompt-константу**

В `web/server.py` рядом с другими `_*_SYSTEM` (около строки 895)
добавить:

```python
_GENERAL_SYSTEM = (
    "Ты — полезный ассистент. Отвечай по делу и кратко. Если пользователь "
    "просит помочь с задачей, для которой пригодилась бы база знаний — "
    "напомни, что в чате есть переключатель режима RAG и его можно "
    "включить или поставить в Auto."
)
```

- [ ] **Step 2: Расширить `_generate_direct_answer` для intent="general"**

В `web/server.py` в функции `_generate_direct_answer` (около строки 928)
дописать ветку для `general` сразу после `other`:

```python
    if intent == "chitchat":
        system = _CHITCHAT_SYSTEM
    elif intent == "meta":
        system = _META_SYSTEM
    elif intent == "other":
        system = _OTHER_SYSTEM
    elif intent == "general":
        system = _GENERAL_SYSTEM
    else:
        system = _CHITCHAT_SYSTEM
```

Также в блоке про «meta даём явную историю» дополнительно для `general`
передавать историю в качестве контекста (LLM лучше отвечает с памятью):

```python
    if intent in ("meta", "general") and history:
        hist_block = "\n".join(
            f"{m.role}: {m.content}" for m in history
        )
        user_content = (
            f"История диалога:\n{hist_block}\n\n"
            f"Текущий вопрос пользователя: {query}"
        )
    else:
        user_content = query
```

- [ ] **Step 3: Добавить поле `bypass_rag` в `AskRequest`**

В `class AskRequest` (около строки 180) добавить рядом с `auto_route`:

```python
    # Принудительный bypass RAG: пользователь явно выбрал режим без
    # retrieval. В отличие от auto_route, тут не запускается классификатор —
    # сразу идём в _generate_direct_answer с intent="general".
    bypass_rag: bool = False
```

- [ ] **Step 4: Ветка bypass в `ask()`**

В функции `ask(req: AskRequest)` (около строки 1102) до router-блока
добавить ранний возврат:

```python
    # 0. Bypass RAG: пользователь явно выбрал режим без retrieve.
    if req.bypass_rag:
        history = (
            _history_store.get_recent(req.chat_id, limit=8)
            if req.chat_id and _history_store else []
        )
        answer = _generate_direct_answer(
            req.query, history, intent="general", stream=False,
        )
        explain = _empty_explain(chat_id=req.chat_id)
        _save_chat_messages(
            req.chat_id, req.query, answer,
            explain=explain.model_dump(),
        )
        return AskResponse(
            chunks=[], prompt="", answer=answer, explain=explain,
        )
```

- [ ] **Step 5: Ветка bypass в `ask_stream()`**

В функции `ask_stream(...)` добавить параметр и ранний возврат. Около
строки 1201, в сигнатуре функции (после `auto_route: bool = False`):

```python
    bypass_rag: bool = False,
```

Перед блоком `if auto_route:` добавить:

```python
    if bypass_rag:
        history = (
            _history_store.get_recent(chat_id, limit=8)
            if chat_id and _history_store else []
        )
        explain = _empty_explain(chat_id=chat_id)

        def bypass_event_source() -> Iterator[str]:
            yield _sse_event("meta", {
                "chunks": [],
                "prompt": "",
                "explain": explain.model_dump(),
            })
            full = []
            for piece in _generate_direct_answer(
                query, history, intent="general", stream=True,
            ):
                full.append(piece)
                yield _sse_event("token", {"text": piece})
            _save_chat_messages(
                chat_id, query, "".join(full),
                explain=explain.model_dump(),
            )
            yield _sse_event("done", {})

        return StreamingResponse(
            bypass_event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
```

- [ ] **Step 6: Проверка ветки `ask` через curl (POST)**

Run (предполагается, что uvicorn перезагрузился):

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Расскажи анекдот про программистов","bypass_rag":true}' \
  | python -m json.tool | head -40
```

Expected: ответ JSON с `chunks: []`, `prompt: ""`, ненулевой `answer`,
и `"rag_skipped": true` внутри `explain`.

- [ ] **Step 7: Проверка ветки `ask_stream` через curl (SSE)**

Run:

```bash
curl -s -N "http://localhost:8000/api/ask/stream?query=Скажи%20привет&bypass_rag=true" | head -20
```

Expected: первое событие `event: meta` с пустыми `chunks/prompt`, затем
несколько `event: token` с кусками текста, и в конце `event: done`.

- [ ] **Step 8: Регресс — старый клиент не сломан**

Run:

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Что такое HNSW?","top_k":3}' \
  | python -m json.tool | head -10
```

Expected: обычный RAG-ответ с `chunks` непустым (поведение прежнее,
поскольку `bypass_rag` по умолчанию `false`).

- [ ] **Step 9: Commit**

```bash
git add web/server.py
git commit -m "$(cat <<'EOF'
feat(server): add bypass_rag flag and general direct-answer mode

bypass_rag=true пропускает retrieve и router, отвечает через LLM с
новым _GENERAL_SYSTEM-промптом. Подаётся история чата если она есть.
В UI это будет ручной режим "RAG: Off" в composer'е.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Frontend types — добавить `RagMode`

**Files:**
- Modify: `frontend/src/lib/types.ts`

- [ ] **Step 1: Добавить тип `RagMode`**

В `frontend/src/lib/types.ts` около других экспортов типов (после `SearchMode`):

```ts
export type RagMode = "auto" | "on" | "off";
```

- [ ] **Step 2: Добавить поле в `RetrievalSettings`**

В интерфейс `RetrievalSettings` добавить `rag_mode`:

```ts
export interface RetrievalSettings {
  top_k: number;
  min_similarity: number;
  search_mode: SearchMode;
  rerank: boolean;
  decompose: boolean;
  rerank_per_subquery: boolean;
  mmr: boolean;
  mmr_lambda: number;
  min_rerank_score: number;
  expand_context: boolean;
  expand_radius: number;
  rewrite: boolean;
  rewrite_n: number;
  auto_route: boolean;
  streaming: boolean;
  rag_mode: RagMode;
}
```

- [ ] **Step 3: Добавить дефолт**

В `DEFAULT_SETTINGS` добавить последним полем:

```ts
  rag_mode: "auto",
```

- [ ] **Step 4: Проверка типов**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибки в файлах, где `RetrievalSettings` используется
неполно — `settings.ts` (пресеты не содержат `rag_mode` — но это
optional partial). Если ошибок в `types.ts` нет — двигаемся дальше.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/types.ts
git commit -m "$(cat <<'EOF'
feat(types): add RagMode to RetrievalSettings

Поле rag_mode со значениями "auto" | "on" | "off". Дефолт "auto".
Маппится на auto_route/bypass_rag при отправке на бэк (отдельный шаг).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Frontend store — миграция и пресеты

**Files:**
- Modify: `frontend/src/stores/settings.ts`

- [ ] **Step 1: Поднять версию persist и добавить миграцию**

В `frontend/src/stores/settings.ts` в конфиге `persist` (последний
аргумент `create()`) заменить `{ name: "rag-frontend-settings-v1" }` на:

```ts
    {
      name: "rag-frontend-settings-v1",
      version: 2,
      migrate: (persistedState: any, version: number) => {
        if (!persistedState) return persistedState;
        if (version < 2) {
          // Старые записи не знают про rag_mode — дефолт "auto".
          const fixDef = (s: any) =>
            s && s.rag_mode === undefined ? { ...s, rag_mode: "auto" } : s;
          return {
            ...persistedState,
            default: fixDef(persistedState.default) ?? persistedState.default,
            perChat: Object.fromEntries(
              Object.entries(persistedState.perChat ?? {}).map(
                ([k, v]) => [k, fixDef(v)],
              ),
            ),
          };
        }
        return persistedState;
      },
    },
```

- [ ] **Step 2: Обновить пресеты**

В блоке `PRESETS` (в этом же файле) в каждом из трёх пресетов добавить
поле `rag_mode: "auto"`. Например:

```ts
  fast: {
    search_mode: "hybrid",
    rerank: false,
    decompose: false,
    rerank_per_subquery: false,
    mmr: false,
    rewrite: false,
    expand_context: false,
    min_rerank_score: 0,
    auto_route: true,
    streaming: true,
    rag_mode: "auto",
  },
```

Сделать то же самое для `balanced` и `thorough`.

- [ ] **Step 3: Проверить тип всего файла**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок не появляется. (Если появляется — поправить место,
где `RetrievalSettings` использовался как полный тип без `rag_mode`.)

- [ ] **Step 4: Smoke check — store в браузере**

В браузере (страница уже открыта на http://localhost:3000):
1. Открыть DevTools → Application → Local Storage.
2. Найти ключ `rag-frontend-settings-v1` (если есть — значит уже
   запускали раньше).
3. Перезагрузить страницу.
4. Снова посмотреть `rag-frontend-settings-v1` — поля `default` и
   все `perChat[id]` должны теперь содержать `rag_mode: "auto"`.

(Если localStorage пустой — это нормально, миграция отработает в
будущем при первой записи.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/stores/settings.ts
git commit -m "$(cat <<'EOF'
feat(settings): migrate persisted store to v2 with rag_mode field

Старые записи без rag_mode получают "auto". Пресеты fast/balanced/thorough
тоже теперь содержат явный rag_mode для согласованности.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Frontend api — маппинг `rag_mode` в API-вызовы

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Добавить helper для маппинга**

В `frontend/src/lib/api.ts` перед объектом `api = { ... }` добавить
приватный helper:

```ts
/**
 * Маппинг ragMode из UI в два независимых флага бэка.
 *  - "auto": router работает как обычно;
 *  - "on": router выключен → RAG всегда;
 *  - "off": bypass_rag=true → бэк пропускает retrieve.
 */
function ragModeToFlags(mode: RetrievalSettings["rag_mode"]) {
  switch (mode) {
    case "on":  return { auto_route: false, bypass_rag: false };
    case "off": return { auto_route: false, bypass_rag: true };
    case "auto":
    default:    return { auto_route: true,  bypass_rag: false };
  }
}
```

- [ ] **Step 2: Использовать helper в `ask()`**

Заменить в `ask()` строку `auto_route: settings.auto_route,` на:

```ts
        ...ragModeToFlags(settings.rag_mode),
```

И удалить отдельную строку `auto_route` (она теперь приходит из spread'а).

- [ ] **Step 3: Использовать helper в `buildStreamUrl()`**

В `buildStreamUrl()` заменить строку `auto_route: String(settings.auto_route),`
на две строки:

```ts
      auto_route: String(ragModeToFlags(settings.rag_mode).auto_route),
      bypass_rag: String(ragModeToFlags(settings.rag_mode).bypass_rag),
```

(Можно вынести в переменную для DRY, но повторение helper'а тут читается
яснее.)

- [ ] **Step 4: Проверка типов**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок нет.

- [ ] **Step 5: Smoke check через DevTools Network**

В браузере открыть Network tab, отправить любое сообщение в чат.
Найти запрос на `/api/ask` или `/api/ask/stream`. Проверить:
- В payload (или query string) есть `auto_route` и `bypass_rag`.
- При дефолте они `true` и `false` соответственно.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "$(cat <<'EOF'
feat(api): map rag_mode to auto_route + bypass_rag in api client

Введён helper ragModeToFlags(). Обе функции — ask() и buildStreamUrl() —
теперь шлют производные флаги, а не auto_route напрямую.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: UI primitive — shadcn-обёртка над radix-popover

**Files:**
- Create: `frontend/src/components/ui/popover.tsx`

- [ ] **Step 1: Создать обёртку**

Создать `frontend/src/components/ui/popover.tsx` (стандартный shadcn-шаблон):

```tsx
"use client";

import * as React from "react";
import * as PopoverPrimitive from "@radix-ui/react-popover";
import { cn } from "@/lib/utils";

const Popover = PopoverPrimitive.Root;
const PopoverTrigger = PopoverPrimitive.Trigger;

const PopoverContent = React.forwardRef<
  React.ElementRef<typeof PopoverPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content>
>(({ className, align = "start", sideOffset = 6, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      align={align}
      sideOffset={sideOffset}
      className={cn(
        "z-50 w-64 rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md outline-none",
        "data-[state=open]:animate-in data-[state=closed]:animate-out",
        "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
        "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
        className,
      )}
      {...props}
    />
  </PopoverPrimitive.Portal>
));
PopoverContent.displayName = PopoverPrimitive.Content.displayName;

export { Popover, PopoverTrigger, PopoverContent };
```

- [ ] **Step 2: Проверка**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок нет (`@radix-ui/react-popover` уже в `package.json`).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ui/popover.tsx
git commit -m "$(cat <<'EOF'
feat(ui): add Popover primitive (shadcn-style wrapper over radix)

Тонкая обёртка вокруг существующей зависимости @radix-ui/react-popover.
Нужна под RagModeChip; других сценариев пока нет.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Компонент `RagModeChip`

**Files:**
- Create: `frontend/src/components/chat/rag-mode-chip.tsx`

- [ ] **Step 1: Создать компонент**

Создать `frontend/src/components/chat/rag-mode-chip.tsx`:

```tsx
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
  const cur = META[value];
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
          const selected = m === value;
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
```

- [ ] **Step 2: Проверка компиляции**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок нет.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/chat/rag-mode-chip.tsx
git commit -m "$(cat <<'EOF'
feat(chat): add RagModeChip component (Auto/On/Off)

Кнопка-popover с тремя вариантами режима RAG. Управляется снаружи через
value/onChange — не лезет в store сам.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Встроить chip в Composer и ChatView

**Files:**
- Modify: `frontend/src/components/chat/composer.tsx`
- Modify: `frontend/src/components/chat/chat-view.tsx`

- [ ] **Step 1: Расширить пропсы `Composer`**

В `frontend/src/components/chat/composer.tsx` обновить `interface Props`:

```ts
import type { RagMode } from "@/lib/types";
import { RagModeChip } from "./rag-mode-chip";

interface Props {
  onSend: (text: string) => void;
  onStop?: () => void;
  onOpenSettings?: () => void;
  busy?: boolean;
  placeholder?: string;
  ragMode: RagMode;
  onRagModeChange: (v: RagMode) => void;
}
```

И в сигнатуре функции добавить новые параметры:

```ts
export function Composer({
  onSend, onStop, onOpenSettings, busy, placeholder,
  ragMode, onRagModeChange,
}: Props) {
```

- [ ] **Step 2: Вставить chip перед иконкой Settings**

В JSX-разметке, в `<div className="flex items-end gap-2 ...">`, перед
условным рендером кнопки Settings добавить chip первым:

```tsx
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 focus-within:border-primary/60 transition-colors">
          <div className="self-center">
            <RagModeChip value={ragMode} onChange={onRagModeChange} />
          </div>
          {onOpenSettings && (
            ...существующая кнопка шестерёнки без изменений...
          )}
```

- [ ] **Step 3: Прокинуть из `ChatView`**

В `frontend/src/components/chat/chat-view.tsx` найти место вызова
`<Composer ... />` (около строки 273) и добавить новые пропсы:

```tsx
      <Composer
        onSend={handleSend}
        onStop={handleStop}
        onOpenSettings={() => setSettingsOpen(true)}
        busy={isStreaming}
        ragMode={settings.rag_mode}
        onRagModeChange={(v) =>
          useSettings.getState().setFor(chatId, { rag_mode: v })
        }
      />
```

- [ ] **Step 4: Проверка типов**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок нет.

- [ ] **Step 5: Smoke check в браузере**

1. Открыть http://localhost:3000.
2. Composer должен показывать chip `[RAG: Auto ▾]` слева от шестерёнки.
3. Клик по chip'у открывает popover со списком трёх вариантов.
4. Выбрать "On" — chip обновился на `[RAG: On ▾]` с иконкой Database и
   акцентным цветом.
5. Reload страницы — chip остаётся в `On` (благодаря persist).
6. Создать новый чат — там chip = `Auto` (свой `perChat`).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/chat/composer.tsx frontend/src/components/chat/chat-view.tsx
git commit -m "$(cat <<'EOF'
feat(chat): wire RagModeChip into Composer and ChatView

Composer получает ragMode/onRagModeChange как пропсы. ChatView читает из
useSettings и обновляет per-chat. UI-состояние видно в строке без открытия
drawer'а.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: MessageBubble — бейдж «RAG не использовался»

**Files:**
- Modify: `frontend/src/components/chat/message-bubble.tsx`

- [ ] **Step 1: Определить условие bypass'a**

В `message-bubble.tsx` после строки `const hasDetails = role === "assistant" && explain != null;` (около строки 112) добавить:

```ts
  // Условие «ручной bypass»: RAG был пропущен, но не из-за router'а
  // (router заполнил бы routed=true и route_intent). Если routed=false и
  // rag_skipped=true — значит пользователь явно выбрал режим Off.
  const isManualBypass =
    role === "assistant" &&
    explain != null &&
    explain.rag_skipped === true &&
    !explain.routed;
```

- [ ] **Step 2: Скрыть pipeline-кнопку в bypass-режиме**

Заменить `const hasDetails = role === "assistant" && explain != null;` на:

```ts
  const hasDetails =
    role === "assistant" && explain != null && !isManualBypass;
```

(`hasDetails` управляет и кнопкой «Sparkles» и рендером
`AnswerDetailsDrawer` — оба отключатся.)

- [ ] **Step 3: Добавить бейдж под bubble**

После закрывающего тега кнопки с `Sparkles` (около строки 199) и перед
закрывающим `</div>` родительского relative-блока добавить:

```tsx
          {isManualBypass && (
            <div className="absolute -bottom-5 left-0 text-[10px] text-muted-foreground">
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-card px-2 py-0.5">
                <span aria-hidden>○</span>
                RAG не использовался — режим обычного чата
              </span>
            </div>
          )}
```

И увеличить нижний отступ родительского `flex gap-3 group` — заменить
classname верхнего div'а с `"flex gap-3 group ..."` на:

```tsx
    <div className={cn("flex gap-3 group", role === "user" ? "flex-row-reverse" : "flex-row", isManualBypass && "mb-5")}>
```

- [ ] **Step 4: Проверка типов**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок нет.

- [ ] **Step 5: Smoke check в браузере**

1. В чате выбрать chip → `Off`.
2. Спросить любое (например, «напиши хокку про вектора»).
3. Ответ приходит без `Sparkles`-кнопки, под bubble — бейдж
   «RAG не использовался — режим обычного чата».
4. Переключить chip → `Auto` и снова что-то спросить — поведение
   возвращается к нормальному (есть Sparkles-кнопка и pipeline).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/chat/message-bubble.tsx
git commit -m "$(cat <<'EOF'
feat(chat): show "RAG не использовался" badge for manual-bypass messages

Условие — explain.rag_skipped && !explain.routed (т.е. это был выбор
пользователя, а не auto-router). Pipeline-кнопка в этом случае скрывается:
показывать там нечего.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: SettingsDrawer — убрать дублирующий switch auto-route

**Files:**
- Modify: `frontend/src/components/settings/settings-drawer.tsx`

- [ ] **Step 1: Удалить блок auto-route**

В `frontend/src/components/settings/settings-drawer.tsx` в секции
«Генерация» (около строк 91-99) удалить блок про auto-route:

```tsx
              <Row
                label="auto-route"
                hint="LLM решает нужен ли RAG (chitchat / meta / other → без поиска)"
              >
                <Switch
                  checked={settings.auto_route}
                  onCheckedChange={(v) => update({ auto_route: v })}
                />
              </Row>
```

Должна остаться только строка про `streaming`.

- [ ] **Step 2: Проверка типов**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: ошибок нет (`settings.auto_route` всё ещё существует в типе —
поле никуда не делось, просто больше не используется в drawer'е; оно
теперь производное от `rag_mode` в api-слое).

- [ ] **Step 3: Smoke check**

Открыть drawer (шестерёнка). В секции «Генерация» должен остаться
только switch `streaming`. Управление RAG-режимом — только через
chip в composer'е.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/settings/settings-drawer.tsx
git commit -m "$(cat <<'EOF'
refactor(settings): remove auto-route switch from drawer

Управление RAG-режимом перенесено в chip composer'a (Auto/On/Off).
Поле auto_route в типах оставлено — теперь оно derived в api-слое из
rag_mode, для обратной совместимости со старыми клиентами.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: End-to-end ручная проверка

Не пишет код — это финальная контрольная проверка. Если что-то не
сходится — фиксим точечным коммитом и повторяем шаг.

- [ ] **Step 1: Сценарий Auto (дефолт)**

Действия:
1. Создать новый чат.
2. Chip показывает `Auto`.
3. Сообщение «привет» → должен сработать auto-router (chitchat), ответ
   без RAG, под bubble — бейдж/иконка отсутствует (нет manual-bypass),
   но pipeline-кнопка Sparkles показывает что route_intent=chitchat.
4. Сообщение «Что такое HNSW?» → RAG включается, видны chunks.

- [ ] **Step 2: Сценарий On**

Действия:
1. Переключить chip → `On`.
2. Сообщение «привет» → теперь должен запуститься RAG (router пропущен),
   chunks непустые, пусть и нерелевантные.

- [ ] **Step 3: Сценарий Off**

Действия:
1. Переключить chip → `Off`.
2. Сообщение «Что такое HNSW?» → RAG не запускается, бейдж под bubble,
   ответ от LLM напрямую (возможно с галлюцинациями — это ожидаемо).
3. Сообщение «А раньше я что спрашивал?» → ответ опирается на историю
   диалога (мы передаём её в _generate_direct_answer).

- [ ] **Step 4: Persist**

Действия:
1. Переключить chip → `On`. Reload страницы.
2. Chip должен остаться в `On`.

- [ ] **Step 5: Per-chat**

Действия:
1. В чате A — chip=`On`.
2. Создать чат B (sidebar → «Новый чат»).
3. В чате B — chip=`Auto` (дефолт нового чата).

- [ ] **Step 6: Финальный коммит-отметка (если были fix'ы)**

Если по ходу e2e всплыли мелкие правки — закоммитить их отдельным
коммитом:

```bash
git commit -m "fix: e2e нюансы по chip-toggle (см. описание)"
```

- [ ] **Step 7: Финальный `git log` для отчёта**

Run:

```bash
git log --oneline 51db9b8..HEAD
```

Expected: коммиты от задач 1-9 (и опционально 10) в порядке.

---

## Self-review checklist (внутренний)

- Поведение всех трёх состояний chip'a покрыто: Auto (Task 4 + сохранённый
  router), On (Task 4 — `auto_route:false, bypass_rag:false`), Off (Task 1
  + Task 4 — `bypass_rag:true`).
- Дефолт `rag_mode: "auto"` (Task 2) и миграция старых записей (Task 3).
- Бэк-маппинг и фронт-маппинг согласованы по именам флагов (`auto_route`,
  `bypass_rag`).
- UI-feedback при bypass'е (Task 8) использует `explain.rag_skipped` и
  `explain.routed` — оба уже существуют в типе `Explain`
  (`frontend/src/lib/types.ts:73,82`).
- `_empty_explain` уже устанавливает `rag_skipped=True` (`web/server.py:1097`),
  а `routed` остаётся дефолтным `False` — наше условие в Task 8 работает.
- Регресс старых клиентов: Task 1 Step 8 проверяет, Task 9 не трогает
  поле `auto_route` в типе — только убирает из UI.
- Все шаги содержат конкретный код / команды / ожидания — без TBD.
