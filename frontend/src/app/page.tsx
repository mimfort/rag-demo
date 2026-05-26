"use client";

import { ChatView } from "@/components/chat/chat-view";

/**
 * Главная — режим «без чата» (stateless).
 * Можно задать вопрос напрямую: при первом вопросе автоматически
 * создастся чат и роутер уведёт на /chat/[id].
 */
export default function HomePage() {
  return <ChatView chatId={null} />;
}
