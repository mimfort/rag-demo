# RAG Studio — Frontend

Production-grade UI для RAG-системы на бэке: чаты (как в ChatGPT/Perplexity),
персональные настройки retrieval на каждый чат, drag&drop загрузка файлов,
визуализация всех шагов pipeline как «думает AI».

## Стек

- **Next.js 15** (App Router) + React 19 + TypeScript
- **Tailwind CSS 3.4** + shadcn-style компоненты (свои, без CLI)
- **TanStack Query v5** — server state, кеширование, инвалидация
- **Zustand** (с persist) — UI state и per-chat настройки в localStorage
- **EventSource API** — стриминг через SSE без библиотек
- **react-markdown** + **remark-gfm** — рендер ответов LLM
- **lucide-react** — иконки
- **sonner** — toasts

## Запуск

```bash
# 1. Поднять бэк-FastAPI на :8000 (из корня проекта)
cd /Users/aleksejzadoroznyj/PycharmProjects/rag
.venv/bin/uvicorn web.server:app --host 0.0.0.0 --port 8000

# 2. Поставить и запустить фронт на :3000
cd frontend
npm install     # (уже сделано)
npm run dev
```

Открыть **http://localhost:3000**.

Все API-запросы идут на относительные пути `/api/*` — Next.js proxy
(см. `next.config.mjs`) перенаправляет их в FastAPI. CORS настраивать
не нужно — браузер видит только same-origin.

## Структура

```
src/
├── app/                          ← Next.js app router
│   ├── layout.tsx                ← корневой layout: providers + sidebar + main
│   ├── page.tsx                  ← /  — режим «без чата» (stateless)
│   ├── chat/[chatId]/page.tsx    ← /chat/{id} — активный чат
│   └── globals.css               ← tailwind + custom стили
├── components/
│   ├── chat/
│   │   ├── chat-view.tsx         ← главный экран: список сообщений + composer
│   │   ├── composer.tsx          ← поле ввода + send/stop кнопки
│   │   ├── message-bubble.tsx    ← один пузырь (user/assistant)
│   │   └── pipeline-steps.tsx    ← раскрывающийся «think aloud» с шагами
│   ├── sidebar/
│   │   ├── sidebar.tsx           ← левая колонка (чаты + источники + статус)
│   │   ├── chats-list.tsx        ← список чатов с CRUD
│   │   └── sources-panel.tsx     ← drag&drop + список источников
│   ├── settings/
│   │   └── settings-drawer.tsx   ← правый drawer с retrieval-настройками
│   ├── ui/                       ← shadcn-style примитивы
│   └── providers.tsx             ← QueryClient + TooltipProvider + Toaster
├── lib/
│   ├── api.ts                    ← fetch-обёртка над /api/*
│   ├── streaming.ts              ← EventSource helper для SSE
│   ├── types.ts                  ← TypeScript-типы (зеркало Pydantic)
│   └── utils.ts                  ← cn() + fmtMs()
└── stores/
    ├── chat-ui.ts                ← активный draft (стрим), UI-состояние
    └── settings.ts               ← per-chat retrieval settings + persist
```

## Архитектурные решения

- **Один draft на UI** (а не очередь) — пользователь может слать только
  один вопрос за раз, как в ChatGPT. Кнопка Stop прерывает SSE.
- **Optimistic update**: user-сообщение появляется мгновенно в кэше
  TanStack Query до записи в БД.
- **Streaming через EventSource**, не fetch+ReadableStream — стандартная
  семантика SSE с `event:` именами событий.
- **Per-chat settings**: каждый чат хранит свой набор retrieval-параметров
  (rerank/decompose/mmr/...). Дефолт для «без чата» — общий.
- **Citations**: после прибытия meta-payload рендерим ответ через
  ReactMarkdown с подменой `[N]` на кликабельные чипы — клик скроллит
  и подсвечивает соответствующий чанк в раскрытом блоке pipeline.
- **Sources isolation**: в режиме чата `SourcesPanel` показывает приватные
  источники чата; в режиме «без чата» — общую базу. Чекбокс «грузить в
  этот чат» решает куда улетит файл.

## Расширение под LangGraph

Когда будете подключать LangGraph-агентов на бэке:

1. **`Explain` тип** в `types.ts` — там же будут поля agent-шагов
   (tool_calls, intermediate_steps). Просто дополнить интерфейс.
2. **`PipelineSteps`** — самое место визуализировать LangGraph-узлы
   как такие же step-карточки с иконкой и `details`. Уже умеет
   `decompose`/`rewrite`/`route` — добавить `tool_call` / `reflect`
   тривиально.
3. **API-layer** в `lib/api.ts` — отдельный endpoint типа `/api/agent/run`,
   тот же SSE-формат с дополнительными `event: tool_call`, `event: step`.
4. **Streaming.ts** — `openStream` принимает любые `onXxx` колбэки.
   Расширяется добавлением новых обработчиков.

## Сборка для production

```bash
npm run build
npm run start  # запуск собранного на :3000
```

Можно деплоить на любой Node-хостинг (Vercel, Railway, fly.io).
В `next.config.mjs` rewrites нужно будет заменить на реальный URL бэка
или поднять рядом через docker-compose.
