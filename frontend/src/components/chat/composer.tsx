"use client";

import * as React from "react";
import { Send, StopCircle, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface Props {
  onSend: (text: string) => void;
  onStop?: () => void;
  onOpenSettings?: () => void;
  /** Идёт ли сейчас стрим — если да, показываем кнопку Stop. */
  busy?: boolean;
  /** Подсказка-плейсхолдер. */
  placeholder?: string;
}

export function Composer({ onSend, onStop, onOpenSettings, busy, placeholder }: Props) {
  const [text, setText] = React.useState("");
  const taRef = React.useRef<HTMLTextAreaElement>(null);

  const send = React.useCallback(() => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  }, [text, busy, onSend]);

  /** Auto-resize textarea по содержимому. */
  React.useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [text]);

  return (
    <div className="border-t border-border bg-background p-4">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 focus-within:border-primary/60 transition-colors">
          {onOpenSettings && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onOpenSettings}
              className="h-9 w-9 shrink-0"
              title="Настройки чата"
            >
              <Settings2 className="h-4 w-4" />
            </Button>
          )}
          <Textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={placeholder ?? "Спроси что-нибудь… (Enter — отправить, Shift+Enter — новая строка)"}
            className="flex-1 min-h-[40px] max-h-[200px] resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 px-2 py-2"
            rows={1}
          />
          {busy ? (
            <Button
              variant="destructive"
              size="icon"
              onClick={onStop}
              className="h-9 w-9 shrink-0"
              title="Прервать"
            >
              <StopCircle className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={send}
              disabled={!text.trim()}
              className="h-9 w-9 shrink-0"
              title="Отправить"
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground text-center">
          Enter — отправить · Shift+Enter — перенос · Settings — настройки retrieval для этого чата
        </p>
      </div>
    </div>
  );
}
