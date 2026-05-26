"use client";

import { useParams } from "next/navigation";
import { ChatView } from "@/components/chat/chat-view";

export default function ChatPage() {
  const params = useParams<{ chatId: string }>();
  const chatId = params?.chatId ?? null;
  return <ChatView chatId={chatId} />;
}
