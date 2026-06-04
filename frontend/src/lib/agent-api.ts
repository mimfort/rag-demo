/**
 * Тонкий клиент для /api/agent/* — non-streaming ask, история сообщений,
 * билдер URL для SSE-стрима (EventSource не умеет POST → query-string).
 *
 * Стиль и API_BASE — как в ./api.ts.
 */

import type { AgentAskResponse, AgentMessage } from "./agent-types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return API_BASE + path;
}

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const agentApi = {
  ask: (query: string) =>
    json<AgentAskResponse>(apiUrl("/api/agent/ask"), {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  messages: (limit = 200) =>
    json<AgentMessage[]>(apiUrl(`/api/agent/messages?limit=${limit}`)),

  /** EventSource не поддерживает POST — query улетает в query-string. */
  buildStreamUrl: (query: string) => {
    const params = new URLSearchParams({ query });
    return apiUrl(`/api/agent/ask/stream?${params.toString()}`);
  },

  /** Возобновление turn'а: подтверждение (да/нет + уточнение). GET для EventSource. */
  buildResumeUrl: (opts: { threadId: string; confirmed: boolean; correction?: string }) => {
    const params = new URLSearchParams({
      thread_id: opts.threadId,
      confirmed: String(opts.confirmed),
    });
    if (opts.correction) params.set("correction", opts.correction);
    return apiUrl(`/api/agent/resume/stream?${params.toString()}`);
  },
};
