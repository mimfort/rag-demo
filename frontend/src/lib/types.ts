/**
 * Зеркало Pydantic-моделей бэкенда (web/server.py).
 * Если бэк меняется — синхронизируем руками.
 */

export type SearchMode = "vector" | "text" | "hybrid";
export type RouteIntent = "knowledge" | "chitchat" | "meta" | "other";

export interface Chunk {
  source: string;
  chunk_index: number;
  content: string;
  similarity: number;
  vector_rank: number | null;
  text_rank: number | null;
  text_score: number | null;
  rrf_score: number | null;
  reranker_score: number | null;
  original_rank: number | null;
  mmr_rank: number | null;
}

export interface ScoredChunk {
  source: string;
  chunk_index: number;
  similarity: number;
}

export interface RewriteGroup {
  subquery: string;
  rewrites: string[];
  status: string;
}

export interface Explain {
  // embedding
  embed_model: string;
  embed_dim: number;
  embed_ms: number;
  query_norm: number;
  query_preview: number[];
  // top-1 cosine math
  top_chunk_preview?: number[] | null;
  top_chunk_norm?: number | null;
  top_dot?: number | null;
  top_similarity?: number | null;
  // distribution
  all_scores: ScoredChunk[];
  // thresholds
  min_similarity: number;
  below_threshold: boolean;
  // search
  search_mode: string;
  // rerank
  reranked: boolean;
  rerank_ms?: number | null;
  rerank_mode?: string;
  // decompose
  decomposed: boolean;
  decompose_status?: string;
  subqueries?: string[];
  decompose_ms?: number | null;
  // mmr
  mmr_applied?: boolean;
  mmr_lambda?: number | null;
  mmr_candidates?: number | null;
  // threshold filter
  min_rerank_score: number;
  filtered_by_threshold?: number;
  // expand context
  context_expanded?: boolean;
  expand_radius?: number | null;
  neighbors_added?: number;
  // rewrite
  rewritten?: boolean;
  rewrite_status?: string;
  rewrites?: string[];
  rewrite_ms?: number | null;
  rewrite_groups?: RewriteGroup[];
  // conversation
  chat_id?: string | null;
  standalone_query?: string | null;
  standalone_changed?: boolean;
  standalone_ms?: number | null;
  history_used?: number;
  // router
  routed?: boolean;
  route_intent?: RouteIntent | null;
  route_reason?: string | null;
  route_ms?: number | null;
  route_fallback?: boolean;
  rag_skipped?: boolean;
}

export interface AskResponse {
  chunks: Chunk[];
  prompt: string;
  answer: string;
  explain: Explain;
}

/* ---------- Settings (UI-state, отправляется в API при ask) ---------- */

export interface RetrievalSettings {
  top_k: number;
  min_similarity: number;
  search_mode: SearchMode;
  rerank: boolean;
  decompose: boolean;
  rerank_per_subquery: boolean;
  mmr: boolean;
  mmr_lambda: number;
  min_rerank_score: number;
  expand_context: boolean;
  expand_radius: number;
  rewrite: boolean;
  rewrite_n: number;
  auto_route: boolean;
  streaming: boolean;
}

export const DEFAULT_SETTINGS: RetrievalSettings = {
  top_k: 5,
  min_similarity: 0.5,
  search_mode: "hybrid",
  rerank: true,
  decompose: false,
  rerank_per_subquery: false,
  mmr: false,
  mmr_lambda: 0.5,
  min_rerank_score: 0.05,
  expand_context: false,
  expand_radius: 1,
  rewrite: false,
  rewrite_n: 3,
  auto_route: true,
  streaming: true,
};

/* ---------- Chats ---------- */

export interface Chat {
  id: string;
  title: string;
  created_at: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  created_at: string | null;
  /** Pipeline-снимок (только у assistant-сообщений с RAG). */
  chunks?: Chunk[] | null;
  explain?: Explain | null;
  prompt?: string | null;
}

/* ---------- Sources ---------- */

export interface Source {
  source: string;
  chunks: number;
  created_at: string | null;
  chat_id: string | null;
}

export interface UploadResult {
  source: string;
  chunks: number;
}
