/**
 * Тонкая обёртка над fetch — единая точка обращения к FastAPI-бэку.
 *
 * Все вызовы идут на относительные пути /api/* — Next.js proxy
 * (см. next.config.mjs) перенаправляет их на http://localhost:8000.
 * Это избавляет от CORS и работает одинаково в dev и production.
 */

import type {
  AskResponse,
  Chat,
  ChatMessage,
  RetrievalSettings,
  Source,
  UploadResult,
} from "./types";

/**
 * Базовый URL бэка. По умолчанию пустая строка → relative URL (через
 * Next.js rewrites). Можно задать NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
 * чтобы ходить напрямую (для SSE-стриминга — proxy буферизует токены).
 * CORS на бэке должен быть настроен под текущий origin.
 */
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

/** Превращает относительный путь /api/... в полный URL. */
function api_url(path: string): string {
  return API_BASE + path;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : undefined;
    } catch {
      /* ignore */
    }
    throw new ApiError(
      detail || `HTTP ${res.status}`,
      res.status,
      detail,
    );
  }
  return res.json();
}

/* -------- Health -------- */
export const api = {
  health: () => json<{ ok: boolean; chunks_in_db: number }>(api_url("/api/health")),

  /* -------- Chats -------- */
  listChats: () => json<Chat[]>(api_url("/api/chats")),

  createChat: (title?: string | null) =>
    json<Chat>(api_url("/api/chats"), {
      method: "POST",
      body: JSON.stringify({ title: title ?? null }),
    }),

  deleteChat: (id: string) =>
    json<{ chat_id: string; deleted: boolean; chunks_deleted: number }>(
      api_url(`/api/chats/${encodeURIComponent(id)}`),
      { method: "DELETE" },
    ),

  listMessages: (chatId: string) =>
    json<ChatMessage[]>(
      api_url(`/api/chats/${encodeURIComponent(chatId)}/messages`),
    ),

  /* -------- Sources -------- */
  listSources: (chatId?: string | null) => {
    const path = chatId
      ? `/api/sources?chat_id=${encodeURIComponent(chatId)}`
      : "/api/sources";
    return json<Source[]>(api_url(path));
  },

  deleteSource: (source: string) =>
    json<{ source: string; deleted: number }>(
      api_url(`/api/sources/${encodeURIComponent(source)}`),
      { method: "DELETE" },
    ),

  uploadFile: async (file: File, chatId?: string | null): Promise<UploadResult> => {
    const form = new FormData();
    form.append("file", file);
    const path = chatId
      ? `/api/upload?chat_id=${encodeURIComponent(chatId)}`
      : "/api/upload";
    const res = await fetch(api_url(path), { method: "POST", body: form });
    if (!res.ok) {
      let detail: string | undefined;
      try {
        const body = await res.json();
        detail = typeof body?.detail === "string" ? body.detail : undefined;
      } catch {
        /* ignore */
      }
      throw new ApiError(detail || `HTTP ${res.status}`, res.status, detail);
    }
    return res.json();
  },

  /* -------- Ask (non-streaming) -------- */
  ask: (
    query: string,
    settings: RetrievalSettings,
    chatId?: string | null,
  ) =>
    json<AskResponse>(api_url("/api/ask"), {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: settings.top_k,
        min_similarity: settings.min_similarity,
        search_mode: settings.search_mode,
        rerank: settings.rerank,
        decompose: settings.decompose,
        rerank_per_subquery: settings.rerank_per_subquery,
        mmr: settings.mmr,
        mmr_lambda: settings.mmr_lambda,
        min_rerank_score: settings.min_rerank_score,
        expand_context: settings.expand_context,
        expand_radius: settings.expand_radius,
        rewrite: settings.rewrite,
        rewrite_n: settings.rewrite_n,
        auto_route: settings.auto_route,
        chat_id: chatId ?? null,
      }),
    }),

  /**
   * Streaming-URL для EventSource. SSE не поддерживает POST — параметры
   * улетают в query string.
   */
  buildStreamUrl: (
    query: string,
    settings: RetrievalSettings,
    chatId?: string | null,
  ): string => {
    const params = new URLSearchParams({
      query,
      top_k: String(settings.top_k),
      min_similarity: String(settings.min_similarity),
      search_mode: settings.search_mode,
      rerank: String(settings.rerank),
      decompose: String(settings.decompose),
      rerank_per_subquery: String(settings.rerank_per_subquery),
      mmr: String(settings.mmr),
      mmr_lambda: String(settings.mmr_lambda),
      min_rerank_score: String(settings.min_rerank_score),
      expand_context: String(settings.expand_context),
      expand_radius: String(settings.expand_radius),
      rewrite: String(settings.rewrite),
      rewrite_n: String(settings.rewrite_n),
      auto_route: String(settings.auto_route),
    });
    if (chatId) params.set("chat_id", chatId);
    return api_url(`/api/ask/stream?${params.toString()}`);
  },
};

export { ApiError };
