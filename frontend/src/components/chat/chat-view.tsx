"use client";

/**
 * Главный экран чата. Тут вся «механика»: список сообщений, стрим,
 * сохранение в чате через бэк-API.
 *
 * Сообщения подгружаем через TanStack Query: при смене chatId
 * перезагружаем. После каждого ask делаем invalidate — список сообщений
 * обновится из бэка.
 */

import * as React from "react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { Sparkles, MessageSquare } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "@/components/chat/message-bubble";
import { Composer } from "@/components/chat/composer";
import { SettingsDrawer } from "@/components/settings/settings-drawer";

import { api } from "@/lib/api";
import { openStream } from "@/lib/streaming";
import { useSettings } from "@/stores/settings";
import { useChatUi, contentKeyOf, type DraftAssistant } from "@/stores/chat-ui";

interface Props {
  chatId: string | null;  // null = режим «без чата» (stateless)
}

export function ChatView({ chatId }: Props) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const settings = useSettings((s) => s.getFor(chatId));
  const setSettings = useSettings((s) => s.setFor);
  const handleRagModeChange = React.useCallback(
    (v: typeof settings.rag_mode) => setSettings(chatId, { rag_mode: v }),
    [chatId, setSettings],
  );
  const draft = useChatUi((s) => s.draft);
  const setDraft = useChatUi((s) => s.setDraft);
  const updateDraft = useChatUi((s) => s.updateDraft);
  const setLastAnswer = useChatUi((s) => s.setLastAnswer);
  const lastAnswer = useChatUi((s) => s.lastAnswers[chatId ?? "none"]);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const bottomRef = React.useRef<HTMLDivElement>(null);

  // Сообщения чата — только когда chatId есть.
  const messagesQuery = useQuery({
    queryKey: ["messages", chatId],
    queryFn: () => (chatId ? api.listMessages(chatId) : Promise.resolve([])),
    enabled: chatId !== null,
  });

  // Если активен draft, но для другого чата — не показываем его здесь.
  const activeDraft: DraftAssistant | null =
    draft && draft.chatId === chatId ? draft : null;

  // Авто-скролл к низу при новых сообщениях / новых токенах.
  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messagesQuery.data?.length, activeDraft?.text]);

  /** Создать новый чат (вызывается при первом сообщении в режиме «без чата»). */
  const createChat = useMutation({
    mutationFn: async (firstQuery: string) => {
      // Title = первые ~60 символов вопроса. Бэк это и сам сделает,
      // но клиентский preview быстрее.
      const title = firstQuery.trim().slice(0, 60) || "Новый чат";
      return api.createChat(title);
    },
    onSuccess: (chat) => {
      queryClient.invalidateQueries({ queryKey: ["chats"] });
      router.push(`/chat/${chat.id}`);
    },
  });

  /** Подготовка и запуск SSE / REST. */
  const startStream = React.useCallback(
    (text: string, targetChatId: string) => {
      // Стоп предыдущего стрима если был.
      draft?.abort?.();

      // Оптимистично кладём пользовательское сообщение в кэш —
      // оно появится сразу, без ожидания записи в БД.
      queryClient.setQueryData(
        ["messages", targetChatId],
        (old: any) => [
          ...(old ?? []),
          { role: "user", content: text, created_at: new Date().toISOString() },
        ],
      );

      // Создаём draft — там будут накапливаться токены ответа.
      const initial: DraftAssistant = {
        chatId: targetChatId,
        userQuery: text,
        text: "",
        chunks: null,
        explain: null,
        prompt: null,
        startedAt: Date.now(),
        finishedAt: null,
        abort: null,
      };
      setDraft(initial);

      const persistAnswer = (
        finalText: string,
        chunks: typeof initial.chunks,
        explain: typeof initial.explain,
        prompt: string | null,
      ) => {
        // Сохраняем «последний ответ» в чате — drawer деталей будет жить
        // даже после того как draft почистится.
        if (explain && chunks) {
          setLastAnswer(targetChatId, {
            chunks,
            explain,
            prompt: prompt ?? "",
            contentKey: contentKeyOf(finalText),
          });
        }
      };

      if (settings.streaming) {
        const url = api.buildStreamUrl(text, settings, targetChatId);
        const stop = openStream(url, {
          onMeta: (m) =>
            updateDraft({
              chunks: m.chunks,
              explain: m.explain,
              prompt: m.prompt,
            }),
          onToken: (piece) => {
            // Аккуратно дописываем к тексту.
            const cur = useChatUi.getState().draft;
            if (!cur) return;
            updateDraft({ text: cur.text + piece });
          },
          onDone: () => {
            const cur = useChatUi.getState().draft;
            if (cur) {
              persistAnswer(cur.text, cur.chunks, cur.explain, cur.prompt);
            }
            updateDraft({ finishedAt: Date.now() });
            queryClient.invalidateQueries({
              queryKey: ["messages", targetChatId],
            });
            // через короткую задержку сбрасываем draft — к этому моменту
            // listMessages уже вернёт ответ от бэка, а lastAnswer уже
            // прикреплён к последнему сообщению через persistAnswer.
            setTimeout(() => {
              const c = useChatUi.getState().draft;
              if (c?.chatId === targetChatId && c.finishedAt) {
                setDraft(null);
              }
            }, 500);
          },
          onError: (err) => {
            toast.error("Streaming прервался: " + err.message);
            setDraft(null);
          },
        });
        updateDraft({ abort: stop });
      } else {
        // Non-streaming через REST.
        api
          .ask(text, settings, targetChatId)
          .then((res) => {
            updateDraft({
              text: res.answer,
              chunks: res.chunks,
              explain: res.explain,
              prompt: res.prompt,
              finishedAt: Date.now(),
            });
            persistAnswer(res.answer, res.chunks, res.explain, res.prompt);
            queryClient.invalidateQueries({
              queryKey: ["messages", targetChatId],
            });
            setTimeout(() => {
              const c = useChatUi.getState().draft;
              if (c?.chatId === targetChatId && c.finishedAt) {
                setDraft(null);
              }
            }, 500);
          })
          .catch((err) => {
            toast.error("Ошибка: " + err.message);
            setDraft(null);
          });
      }
    },
    [draft, settings, queryClient, setDraft, updateDraft, setLastAnswer],
  );

  /** Отправить вопрос: streaming или non-streaming в зависимости от настроек. */
  const handleSend = React.useCallback(
    async (text: string) => {
      // Если работаем «без чата» — создаём чат и переходим в него,
      // первое сообщение отправит уже там (см. эффект ниже через query).
      if (chatId === null) {
        const chat = await createChat.mutateAsync(text);
        // Стартуем стрим сразу — но через новый chatId.
        startStream(text, chat.id);
        return;
      }
      startStream(text, chatId);
    },
    // startStream закрывается над settings, поэтому обязан быть в deps —
    // иначе после смены chip'а handleSend будет использовать stale-функцию
    // с устаревшим rag_mode и слать неправильные флаги на бэк.
    [chatId, createChat, startStream],
  );

  const handleStop = React.useCallback(() => {
    draft?.abort?.();
    setDraft(null);
  }, [draft, setDraft]);

  const messages = messagesQuery.data ?? [];
  const isStreaming = activeDraft !== null && activeDraft.finishedAt === null;

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
          {messages.length === 0 && !activeDraft && (
            <EmptyState />
          )}

          {messages.map((m, i) => {
            // Берём pipeline-данные либо из БД (сохранены при сообщении),
            // либо как fallback — из in-memory кеша lastAnswer для
            // самого свежего ответа (на случай если данные в БД сохранятся
            // чуть позже из-за гонки с invalidate).
            const isLastAssistant =
              m.role === "assistant" &&
              i === messages.length - 1 &&
              !activeDraft;
            const lastAnswerMatches =
              isLastAssistant &&
              lastAnswer &&
              contentKeyOf(m.content) === lastAnswer.contentKey;

            const explain = m.explain ?? (lastAnswerMatches ? lastAnswer.explain : null);
            const chunks = m.chunks ?? (lastAnswerMatches ? lastAnswer.chunks : []);
            const prompt = m.prompt ?? (lastAnswerMatches ? lastAnswer.prompt : null);

            return (
              <MessageBubble
                key={`${m.created_at}-${i}`}
                role={m.role}
                content={m.content}
                explain={explain}
                chunks={chunks ?? []}
                prompt={prompt}
              />
            );
          })}

          {/* Draft: показываем «думающий» assistant под последним user-сообщением. */}
          {activeDraft && (
            <MessageBubble
              role="assistant"
              content={activeDraft.text}
              explain={activeDraft.explain}
              chunks={activeDraft.chunks ?? []}
              prompt={activeDraft.prompt}
              streaming={isStreaming}
            />
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <Composer
        onSend={handleSend}
        onStop={handleStop}
        onOpenSettings={() => setSettingsOpen(true)}
        busy={isStreaming}
        ragMode={settings.rag_mode}
        onRagModeChange={handleRagModeChange}
      />

      <SettingsDrawer
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        chatId={chatId}
      />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center text-center py-16 gap-3">
      <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
        <Sparkles className="h-6 w-6 text-primary" />
      </div>
      <h2 className="text-xl font-semibold">Чат с базой знаний</h2>
      <p className="text-sm text-muted-foreground max-w-md">
        Задай вопрос — система найдёт ответ в загруженных документах.
        Переключателем RAG в строке ввода можно выбрать Auto (LLM сам решает),
        On (всегда искать) или Off (обычный чат без базы).
      </p>
      <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
        <MessageSquare className="h-3.5 w-3.5" />
        Каждое сообщение раскрывается в подробности pipeline
      </div>
    </div>
  );
}
