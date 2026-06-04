"use client";

import * as React from "react";
import { Check, X, HelpCircle, Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ClarifyData } from "@/lib/agent-types";

export function ClarifyPrompt({
  data,
  onConfirm,
  onReject,
  disabled,
}: {
  data: ClarifyData;
  onConfirm: () => void;
  onReject: (correction: string) => void;
  disabled?: boolean;
}) {
  const [rejecting, setRejecting] = React.useState(false);
  const [correction, setCorrection] = React.useState("");

  const submitReject = () => {
    const c = correction.trim();
    if (!c) return;
    onReject(c);
    setRejecting(false);
    setCorrection("");
  };

  return (
    <div className="rounded-2xl border border-amber-400/40 bg-amber-50/50 dark:bg-amber-950/20 px-4 py-3 space-y-3">
      <div className="flex items-start gap-2">
        <HelpCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
        <div className="text-sm">
          <span className="text-muted-foreground">Вы имели в виду:</span>{" "}
          <span className="font-medium">«{data.interpretation}»</span>
        </div>
      </div>

      {!rejecting ? (
        <div className="flex gap-2">
          <Button size="sm" onClick={onConfirm} disabled={disabled} className="gap-1">
            <Check className="h-3.5 w-3.5" /> Да, верно
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setRejecting(true)}
            disabled={disabled}
            className="gap-1"
          >
            <X className="h-3.5 w-3.5" /> Нет
          </Button>
        </div>
      ) : (
        <div className="flex items-end gap-2">
          <Textarea
            autoFocus
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitReject();
              }
            }}
            placeholder="Что вы имели в виду?"
            className="flex-1 min-h-[38px] max-h-[120px] resize-none"
            rows={1}
          />
          <Button size="icon" onClick={submitReject} disabled={disabled || !correction.trim()} className="h-9 w-9 shrink-0">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
