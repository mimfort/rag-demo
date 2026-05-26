/**
 * UI-состояние чатов: какой открыт, текущая стрим-сессия, оптимистичные
 * сообщения «в полёте» (assistant-черновик который наполняется
 * SSE-токенами и инфой о pipeline) + кеш «последнего ответа» каждого чата
 * чтобы открыть drawer деталей.
 *
 * Не хранит данные с бэка — для них используется TanStack Query.
 */

import { create } from "zustand";
import type { Chunk, Explain } from "@/lib/types";

/** Сообщение, которое сейчас собирается через streaming. */
export interface DraftAssistant {
  chatId: string | null;          // null = «без чата» (stateless)
  userQuery: string;              // что пользователь спросил
  text: string;                   // накопленный текст ответа
  chunks: Chunk[] | null;         // когда придёт meta — заполняется
  explain: Explain | null;
  prompt: string | null;          // user-prompt который ушёл в LLM
  startedAt: number;
  finishedAt: number | null;
  abort: (() => void) | null;     // функция для прерывания SSE
}

/**
 * Детали последнего ответа в чате — для drawer'а «как я дошёл до этого».
 * Не сохраняются на бэке, поэтому доступны только пока вкладка открыта.
 * Когда пользователь перезагрузит страницу — детали для старых сообщений
 * пропадут, но текущий и последний — останутся.
 */
export interface AnswerDetails {
  chunks: Chunk[];
  explain: Explain;
  prompt: string;
  /** Хэш ответа (короткий префикс контента) — чтобы сопоставить с message. */
  contentKey: string;
}

/** Ключ для lastAnswers map: chatId или "none" для stateless. */
const detailsKey = (chatId: string | null) => chatId ?? "none";

interface ChatUiState {
  /** Текущая стрим-сессия (одна на весь UI). null = бот не работает. */
  draft: DraftAssistant | null;
  setDraft: (d: DraftAssistant | null) => void;
  updateDraft: (patch: Partial<DraftAssistant>) => void;

  /**
   * Кеш «последний ответ → детали» для каждого чата (включая stateless).
   * Используется чтобы показать drawer для последнего assistant-message
   * после того как draft стёрт (поверх данных из БД).
   */
  lastAnswers: Record<string, AnswerDetails>;
  setLastAnswer: (chatId: string | null, details: AnswerDetails) => void;
  getLastAnswer: (chatId: string | null) => AnswerDetails | undefined;
}

export const useChatUi = create<ChatUiState>((set, get) => ({
  draft: null,
  setDraft: (d) => set({ draft: d }),
  updateDraft: (patch) => {
    const cur = get().draft;
    if (!cur) return;
    set({ draft: { ...cur, ...patch } });
  },

  lastAnswers: {},
  setLastAnswer: (chatId, details) =>
    set((s) => ({
      lastAnswers: { ...s.lastAnswers, [detailsKey(chatId)]: details },
    })),
  getLastAnswer: (chatId) => get().lastAnswers[detailsKey(chatId)],
}));

/** Короткий ключ контента — первые 80 символов после whitespace-нормализации. */
export function contentKeyOf(text: string): string {
  return text.trim().slice(0, 80);
}
