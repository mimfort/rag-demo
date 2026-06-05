/**
 * Типы LangGraph-агента — зеркало Pydantic-моделей бэка
 * (см. web/server.py и agent/*).
 */

export type AgentEventType =
  | "clarify"
  | "node_start"
  | "tool_call"
  | "tool_result"
  | "verify"
  | "final_answer"
  | "done"
  | "error";

export interface ClarifyData {
  thread_id: string;
  interpretation: string;
  original: string;
  round: number;
}

export interface TraceEvent {
  type: AgentEventType;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface AgentAskResponse {
  answer: string;
  trace: TraceEvent[];
  iterations: number;
}

export interface AgentMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  trace: TraceEvent[] | null;
  created_at: string;
}
