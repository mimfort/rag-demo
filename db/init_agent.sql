-- =============================================================================
-- init_agent.sql — таблица для LangGraph-агента (Spec 1).
-- Выполняется автоматически Docker'ом при первом старте контейнера
-- (docker-entrypoint-initdb.d), либо вручную psql'ом для running БД.
-- =============================================================================

CREATE TABLE IF NOT EXISTS agent_messages (
    id          BIGSERIAL PRIMARY KEY,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    trace       JSONB,                -- массив TraceEvent для assistant; для user — NULL
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_messages_created_at_idx
    ON agent_messages (created_at);
