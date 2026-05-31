# Spec 1 — RAG mode chip-toggle

Дата: 2026-05-27
Статус: предложение, ожидает финального ревью

## Зачем

В composer'е сейчас нет явного способа управлять тем, использует ли система
RAG. Косвенно есть `auto_route` — он автоматически классифицирует запрос и
пропускает RAG для chitchat/meta/other. Но пользователь не может:

- принудительно прогнать запрос через RAG (если auto-router ошибётся);
- быстро отключить RAG и поговорить «как с обычным ChatGPT» в этом чате;
- увидеть текущее поведение, не открывая drawer настроек.

Этот спек добавляет **chip-toggle в composer** с тремя состояниями
`Auto / On / Off` — небольшая, изолированная фича. После её появления
дальше идут отдельные спеки на улучшения RAG (citations + self-check) и
на source filter + eval gate.

## Поведение

Chip размещается в composer'е слева от шестерёнки. Три состояния:

| Состояние   | Поведение                                                       |
|-------------|-----------------------------------------------------------------|
| `Auto`      | Текущее поведение: `auto_route` классифицирует, RAG используется только для intent=knowledge. |
| `On`        | Принудительный RAG для каждого сообщения, `auto_route` игнорируется. |
| `Off`       | RAG не запускается, ответ строится напрямую через LLM с general system-prompt'ом и историей чата. Pipeline-блоки в UI скрываются. |

Дефолт у нового чата — `Auto`. Состояние сохраняется per-chat в
`useSettings` (как и остальные настройки retrieval). Состояние видно
прямо в строке без необходимости открывать drawer.

## Типы и API

### Frontend

В `frontend/src/lib/types.ts`:

```ts
export type RagMode = "auto" | "on" | "off";

export interface RetrievalSettings {
  // ...
  rag_mode: RagMode;
}

export const DEFAULT_SETTINGS: RetrievalSettings = {
  // ...
  rag_mode: "auto",
};
```

### Backend

В `web/server.py`:

```python
class AskRequest(BaseModel):
    # ...
    bypass_rag: bool = False  # явный override от UI
```

### Маппинг UI → API

`rag_mode` выводится в два уже-существующих/новых флага бэка:

| `rag_mode` | `auto_route` | `bypass_rag` | Эффект                           |
|------------|--------------|--------------|----------------------------------|
| `"auto"`   | `true`       | `false`      | Как сейчас.                      |
| `"on"`     | `false`      | `false`      | Router выключен — всегда RAG.    |
| `"off"`    | `false`      | `true`       | Бэк пропускает retrieve.         |

`auto_route` остаётся в типах и в API (для обратной совместимости с
existing вызовами и тестами), но в UI становится **производным** от
`rag_mode`. Drawer с детальными настройками больше не показывает
отдельный switch для `auto_route` — он скрыт за chip'ом.

## UI

Новый компонент `src/components/chat/rag-mode-chip.tsx`:

- shadcn `DropdownMenu` с `RadioGroup` из трёх пунктов: Auto / On / Off.
- У каждого пункта — короткая подпись-описание.
- Сам chip — `Button variant="ghost" size="sm"` с иконкой и текстом
  текущего состояния. Иконки:
  - Auto → `Sparkles` (lucide), нейтральный цвет;
  - On → `Database`, акцент `text-primary`;
  - Off → `MessageSquareOff`, цвет `text-muted-foreground`.
- Если состояние = `Off`, у chip'а есть subtle outline-вариант чтобы
  было видно «RAG отключён» периферийным зрением.

В `Composer` chip добавляется в flex-row до иконки шестерёнки:

```
[RAG-chip]  [Settings ⚙]  [textarea]  [Send/Stop]
```

В `MessageBubble`: ветка отображения когда сообщение было сгенерировано
с `bypass_rag`. Признак — `explain.rag_skipped === true && explain.routed === false`
(router не вызывался — значит, был bypass от UI). В этом случае:

- скрываем аккордеон «шагов pipeline»;
- под текстом ответа маленький бейдж: «RAG не использовался — режим
  обычного чата».

## Изменения по файлам

### Frontend

| Файл                                            | Что меняется                                                       |
|-------------------------------------------------|--------------------------------------------------------------------|
| `src/lib/types.ts`                              | `RagMode` type, поле в `RetrievalSettings`, в `DEFAULT_SETTINGS`.  |
| `src/lib/api.ts`                                | В `ask()` и `buildStreamUrl()` мапить `rag_mode` → `auto_route` + `bypass_rag`. |
| `src/stores/settings.ts`                        | Миграция: при чтении старых записей без `rag_mode` подставлять `"auto"`. Версия persist'а поднимается до `v2`. |
| `src/components/chat/rag-mode-chip.tsx` (new)   | Новый компонент.                                                   |
| `src/components/chat/composer.tsx`              | Добавить prop `ragMode`/`onRagModeChange` и render chip'а.         |
| `src/components/chat/chat-view.tsx`             | Передать `settings.rag_mode` и `setFor({rag_mode})` в Composer.    |
| `src/components/chat/message-bubble.tsx`        | Ветка «bypass-режим»: скрываем pipeline, рисуем бейдж.             |
| `src/components/settings/settings-drawer.tsx`   | Спрятать switch `auto_route` (он теперь производный). В пресетах `fast/balanced/thorough` добавить `rag_mode: "auto"`. |

### Backend

| Файл               | Что меняется                                                                                                                                                                   |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `web/server.py`    | `AskRequest`: добавить `bypass_rag: bool = False`. В `ask()` и `ask_stream()` early-return ветка `if bypass_rag` до router'а — вызвать `_generate_direct_answer` с новым intent `"general"`. Добавить константу `_GENERAL_SYSTEM`. Расширить `_generate_direct_answer` чтобы понимал `intent="general"`. |

### Backwards compatibility

- Старые чаты в localStorage без `rag_mode` стартуют в `Auto`.
- Старые API-вызовы без `bypass_rag` работают (дефолт `False`).
- `auto_route` остаётся в API — старые клиенты не ломаются.

## Тестирование

### Backend (вручную через curl)

1. `POST /api/ask` с `bypass_rag: true` — ожидаем пустой `chunks`,
   пустой `prompt`, ненулевой `answer`, `explain.rag_skipped === true`.
2. `POST /api/ask` с `bypass_rag: false, auto_route: false` — ожидаем
   обычный RAG-ответ независимо от типа вопроса.
3. `GET /api/ask/stream?...&bypass_rag=true` через `curl --no-buffer` —
   ожидаем `meta` с пустыми `chunks/prompt`, затем стрим токенов, затем
   `done`.
4. Регресс: вызов без `bypass_rag` (старый клиент) — поведение прежнее.

### Frontend (вручную)

1. Поднять dev-сервер: `cd frontend && npm run dev`.
2. В новом чате убедиться, что chip показывает `Auto` и сообщение
   "привет" не запускает RAG (auto-router отрабатывает).
3. Переключить chip → `On`, отправить "привет" — должен запуститься
   RAG (видны chunks). Это проверяет, что `auto_route` действительно
   выключился.
4. Переключить chip → `Off`, отправить вопрос про базу — RAG не
   запускается, ответ приходит без chunks, под bubble видно бейдж.
5. Reload страницы — chip остаётся в выбранном состоянии (per-chat
   persist).
6. Открыть старый чат, где в localStorage нет `rag_mode` — chip
   должен показать `Auto` без ошибок.

### Регресс RAG-пайплайна

Запустить `python -m evals.runner` (если есть baseline) и убедиться,
что метрики не изменились на baseline-режиме `Auto`. Этот спек не
трогает retrieval-логику, поэтому изменений ждать не следует.

## Out of scope (отдельные спеки)

- Inline citations с подсветкой источников (Spec 2).
- Self-check / answer grader (Spec 2).
- Source filter chips над composer'ом (Spec 3).
- Eval regression gate как часть pre-commit / CI (Spec 3).
- LangGraph-агенты — будут после Spec 1-3.

## Следующий шаг

После согласования этого spec'а — переход к writing-plans skill для
разработки пошагового плана реализации.
