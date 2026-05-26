/**
 * Хранение retrieval-настроек per-chat (и default для «без чата»).
 *
 * Структура:
 *   - default: RetrievalSettings — используется когда нет активного чата
 *   - perChat: { [chatId]: RetrievalSettings } — индивидуальные настройки
 *
 * Сохраняется в localStorage (через persist middleware).
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { DEFAULT_SETTINGS, type RetrievalSettings } from "@/lib/types";

interface SettingsState {
  default: RetrievalSettings;
  perChat: Record<string, RetrievalSettings>;

  /** Получить настройки для конкретного чата (или default если нет). */
  getFor: (chatId: string | null) => RetrievalSettings;

  /** Обновить настройки конкретного чата (или default). */
  setFor: (
    chatId: string | null,
    update: Partial<RetrievalSettings>,
  ) => void;

  /** Заменить целиком (для пресетов). */
  replaceFor: (chatId: string | null, value: RetrievalSettings) => void;

  /** Очистить настройки удалённого чата (вызывается при delete). */
  forget: (chatId: string) => void;
}

export const useSettings = create<SettingsState>()(
  persist(
    (set, get) => ({
      default: DEFAULT_SETTINGS,
      perChat: {},

      getFor: (chatId) => {
        const s = get();
        if (chatId && s.perChat[chatId]) return s.perChat[chatId];
        return s.default;
      },

      setFor: (chatId, update) => {
        if (chatId == null) {
          set((s) => ({ default: { ...s.default, ...update } }));
        } else {
          set((s) => ({
            perChat: {
              ...s.perChat,
              [chatId]: { ...(s.perChat[chatId] ?? s.default), ...update },
            },
          }));
        }
      },

      replaceFor: (chatId, value) => {
        if (chatId == null) {
          set({ default: value });
        } else {
          set((s) => ({
            perChat: { ...s.perChat, [chatId]: value },
          }));
        }
      },

      forget: (chatId) => {
        set((s) => {
          const next = { ...s.perChat };
          delete next[chatId];
          return { perChat: next };
        });
      },
    }),
    { name: "rag-frontend-settings-v1" },
  ),
);

/* -------- Пресеты «быстрых» конфигов -------- */

export const PRESETS: Record<string, Partial<RetrievalSettings>> = {
  /** Минимум — vector-поиск без всего. Быстро. */
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
  },
  /** Сбалансированный — то что обычно стоит по умолчанию. */
  balanced: {
    search_mode: "hybrid",
    rerank: true,
    decompose: false,
    rerank_per_subquery: false,
    mmr: false,
    rewrite: false,
    expand_context: true,
    expand_radius: 1,
    min_rerank_score: 0.05,
    auto_route: true,
    streaming: true,
  },
  /** Максимум качества — все слои на. Медленно. */
  thorough: {
    search_mode: "hybrid",
    rerank: true,
    decompose: true,
    rerank_per_subquery: true,
    mmr: true,
    mmr_lambda: 0.5,
    rewrite: false, // rewrite + decompose очень долго на CPU
    expand_context: true,
    expand_radius: 2,
    min_rerank_score: 0.05,
    auto_route: true,
    streaming: true,
  },
};
