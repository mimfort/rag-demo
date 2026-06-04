/**
 * Ephemeral-store для UI агента: держит draft assistant-сообщения во время
 * SSE-стрима (накапливающийся trace + финальный answer).
 *
 * В отличие от useSettings — без persist: всё живёт в памяти и сбрасывается
 * по завершении/ошибке стрима.
 */

import { create } from "zustand";
import type { TraceEvent, ClarifyData } from "@/lib/agent-types";

/** Draft assistant-сообщения во время стрима (накапливающийся trace). */
export interface AgentDraft {
  userQuery: string;
  trace: TraceEvent[];
  answer: string;       // обновляется на final_answer event
  finished: boolean;
  abort: (() => void) | null;
  pendingClarify: ClarifyData | null;  // карточка подтверждения (null — нет паузы)
}

interface AgentUIState {
  draft: AgentDraft | null;
  setDraft: (d: AgentDraft | null) => void;
  appendTrace: (ev: TraceEvent) => void;
  setAnswer: (text: string) => void;
  setFinished: (v: boolean) => void;
  setPendingClarify: (c: ClarifyData | null) => void;
}

export const useAgentUi = create<AgentUIState>((set) => ({
  draft: null,
  setDraft: (d) => set({ draft: d }),
  appendTrace: (ev) =>
    set((s) => {
      if (!s.draft) return s;
      return { draft: { ...s.draft, trace: [...s.draft.trace, ev] } };
    }),
  setAnswer: (text) =>
    set((s) => (s.draft ? { draft: { ...s.draft, answer: text } } : s)),
  setFinished: (v) =>
    set((s) => (s.draft ? { draft: { ...s.draft, finished: v } } : s)),
  setPendingClarify: (c) =>
    set((s) => (s.draft ? { draft: { ...s.draft, pendingClarify: c } } : s)),
}));
