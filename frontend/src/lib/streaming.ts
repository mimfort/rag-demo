/**
 * Обёртка над EventSource для типизированного SSE.
 *
 * Бэк присылает три типа событий:
 *   event: meta   — данные о retrieve (chunks, prompt, explain)
 *   event: token  — очередной кусок текста ответа
 *   event: done   — конец потока
 */

import type { Chunk, Explain } from "./types";

export interface MetaPayload {
  chunks: Chunk[];
  prompt: string;
  explain: Explain;
}

export interface StreamHandlers {
  onMeta: (meta: MetaPayload) => void;
  onToken: (text: string) => void;
  onDone: () => void;
  onError?: (error: Error) => void;
}

/**
 * Открывает SSE-соединение, парсит события, вызывает колбэки.
 * Возвращает функцию для прерывания потока.
 */
export function openStream(url: string, handlers: StreamHandlers): () => void {
  const es = new EventSource(url);
  let finished = false;

  es.addEventListener("meta", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data) as MetaPayload;
      handlers.onMeta(payload);
    } catch (e) {
      handlers.onError?.(e as Error);
    }
  });

  es.addEventListener("token", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data) as { text: string };
      if (payload?.text) handlers.onToken(payload.text);
    } catch {
      /* пропускаем битый чанк */
    }
  });

  es.addEventListener("done", () => {
    finished = true;
    es.close();
    handlers.onDone();
  });

  es.onerror = () => {
    // У EventSource onerror срабатывает и при нормальном close, и при обрыве.
    // Различаем по readyState.
    if (es.readyState === EventSource.CLOSED) {
      if (!finished) handlers.onDone();
    } else {
      es.close();
      handlers.onError?.(new Error("SSE: соединение прервано"));
    }
  };

  return () => {
    finished = true;
    es.close();
  };
}
