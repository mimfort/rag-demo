/**
 * Типы LangGraph-агента — зеркало Pydantic-моделей бэка
 * (см. rag/api/agent_routes.py и rag/agent/*).
 */

export type AgentEventType =
  | "node_start"
  | "tool_call"
  | "tool_result"
  | "final_answer"
  | "done"
  | "error";

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
