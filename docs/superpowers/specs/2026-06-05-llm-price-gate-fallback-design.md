# Ценовой гейт + fallback-модель + выбор выгодного провайдера (OpenRouter)

Дата: 2026-06-05

## Цель

Три связанные возможности поверх chat-LLM (OpenRouter):

1. **Ценовой гейт.** В `.env` задаются максимально допустимые цены за 1М prompt- и
   completion-токенов. Если цена настроенной модели выше — приложение не стартует.
2. **Fallback-модель.** Если основная модель недоступна/падает — OpenRouter
   автоматически переключается на запасную (нативный механизм `models[]`).
3. **Выбор выгодного провайдера.** Опционально включить `provider.sort` (`price` /
   `throughput` / `latency`), чтобы OpenRouter маршрутизировал к самому дешёвому
   (или быстрому) провайдеру модели.

## Решения (брейншторм)

| Вопрос | Решение |
|---|---|
| Механизм fallback | Нативный OpenRouter: `models: [primary, fallback]` в запросе |
| Поведение при недоступном прайсинге | fail-open: warning + продолжаем |
| Лимит для запасной модели | Да — проверяются обе модели |
| Куда вешать гейт | DRY: в конструкторы chat-клиентов (одна проверка на процесс) |
| Когда проверять цену | На старте (один раз за процесс), не на каждый запрос |
| Имена переменных | `LLM_FALLBACK_MODEL`, `LLM_MAX_PROMPT_PRICE_PER_MTOK`, `LLM_MAX_COMPLETION_PRICE_PER_MTOK`, `LLM_PROVIDER_SORT` |

## Факты об OpenRouter API

- `GET {base}/models` — публичный (auth не обязателен). Каждый элемент `data[]` имеет
  `id` и `pricing` со строковыми `prompt` / `completion` = USD за 1 токен. ×1e6 → за 1М.
  `"0"` = бесплатно.
- Fallback: в теле запроса `models: [primary, fallback, ...]`. При ошибке основной
  (downtime / rate-limit / отказ модерации) OpenRouter идёт по списку. Платится по
  фактически использованной модели (возвращается в `model` ответа). Работает со стримом.
- Provider routing: `provider: {"sort": "price"|"throughput"|"latency"}`. `sort: "price"`
  всегда выбирает самого дешёвого провайдера (без балансировки).

## Конфигурация

Новые переменные `.env` / `.env.example` (все опциональные, пусто = выключено):

```dotenv
# Запасная модель: при недоступности основной OpenRouter переключится на неё.
LLM_FALLBACK_MODEL=
# Ценовой потолок, USD за 1М токенов. Пусто = без ограничения.
LLM_MAX_PROMPT_PRICE_PER_MTOK=
LLM_MAX_COMPLETION_PRICE_PER_MTOK=
# Маршрутизация провайдера: price | throughput | latency. Пусто = дефолт OpenRouter.
LLM_PROVIDER_SORT=
```

`Settings` получает:
- `llm_fallback_model: str` (пустая строка = нет fallback).
- `llm_max_prompt_price: float | None`, `llm_max_completion_price: float | None`
  (через новый хелпер `_env_float_optional`).
- `llm_provider_sort: str` — валидируется на загрузке: пусто или одно из
  `price`/`throughput`/`latency`, иначе `RuntimeError` с понятным сообщением.

## Компоненты

### rag/pricing.py (новый модуль)

Единственная ответственность — узнать цену модели у OpenRouter и проверить против лимита.

- `@dataclass(frozen=True) ModelPrice: prompt_per_mtok: float, completion_per_mtok: float`.
- `fetch_prices(base_url, api_key, *, timeout=10.0, client=None) -> dict[str, ModelPrice]`
  — `GET {base}/models`, парсит `data[].id` + `pricing.prompt`/`pricing.completion`
  (строки → float, ×1e6). Инъекция `client` (httpx.MockTransport) для тестов.
- `class PriceCapExceeded(RuntimeError)` — сообщение: имя модели, её цена, лимит, тип
  токена (prompt/completion).
- `enforce_price_caps_once(settings, *, client=None) -> None` — идемпотентно
  (булев флаг на процесс):
  - если **оба** лимита `None` → ранний выход без сетевых вызовов;
  - иначе собрать список моделей `[llm_model] + ([llm_fallback_model] если задан)`;
  - `fetch_prices(...)`; при исключении фетча → `print` warning, выход (fail-open);
  - для каждой модели: если её id есть в карте цен и цена > лимита →
    `raise PriceCapExceeded`; если id отсутствует → warning (fail-open), пропустить.

### rag/generator.py (ChatGenerator)

- В `__init__` после настройки клиента вызвать `enforce_price_caps_once(settings)`
  (одна проверка на процесс; на пустых лимитах — no-op без сети).
- В `_build_payload` собрать дополнительные поля:
  - если `settings.llm_fallback_model` непустой → `payload["models"] = [model, fallback]`;
  - если `settings.llm_provider_sort` непустой → `payload["provider"] = {"sort": value}`.
  `payload["model"]` остаётся primary. Остальная логика (stream/temperature/messages)
  без изменений — fallback и provider-sort работают и в стриминге.

### agent/llm.py (make_llm)

- Вызвать `enforce_price_caps_once(rag_settings)`.
- Собрать `extra_body`: `{"models": [primary, fallback]}` если fallback задан, плюс
  `{"provider": {"sort": value}}` если provider_sort задан. Передать в
  `ChatOpenAI(..., extra_body=extra_body)` только если непусто (иначе не передавать).

## Поток данных

```
старт процесса → конструируется ChatGenerator / make_llm
                 → enforce_price_caps_once(settings)
                    ├─ лимиты пусты → выход (нет сети)
                    ├─ GET /models → цена > лимита → PriceCapExceeded → процесс падает
                    └─ фетч упал / модели нет → warning → продолжаем
каждый chat-запрос → payload c models[] (fallback) и provider.sort (если заданы)
                     → OpenRouter сам выбирает провайдера и переключает модель при сбое
```

## Обработка ошибок

- `PriceCapExceeded` — на старте, ясное сообщение (модель / цена за 1М / лимит / тип токена).
- Недоступный `/models`, нет модели в списке, не-OpenRouter base_url → warning, fail-open.
- Невалидный `LLM_PROVIDER_SORT` → `RuntimeError` на загрузке конфига.

## Тестирование (httpx.MockTransport, без сети)

- `fetch_prices`: строки `pricing.prompt/completion` → float ×1e6; пустой/`"0"` → 0.0.
- `enforce_price_caps_once`:
  - превышение лимита → `PriceCapExceeded`;
  - в пределах лимита → проходит;
  - модель отсутствует в `/models` → не падает (fail-open);
  - оба лимита пусты → сеть не вызывается (handler с AssertionError);
  - проверяются обе модели (fallback дороже лимита → падает);
  - идемпотентность (повторный вызов не фетчит снова).
- `ChatGenerator._build_payload`:
  - `models` присутствует при заданном fallback, отсутствует без него;
  - `provider` присутствует при заданном sort, отсутствует без него.

## Вне scope

- Учёт реального расхода токенов/бюджета в рантайме (только потолок цены модели).
- Кэширование `/models` между процессами (фетч один раз за процесс).
- Клиентский retry-fallback (используем нативный OpenRouter).
