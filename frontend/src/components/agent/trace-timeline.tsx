"use client";

import * as React from "react";
import type { TraceEvent } from "@/lib/agent-types";
import { TraceStep } from "./trace-step";

interface Props {
  trace: TraceEvent[];
}

export function TraceTimeline({ trace }: Props) {
  if (!trace.length) return null;
  return (
    <div className="flex flex-col gap-0.5 py-2">
      {trace.map((ev, i) => (
        <TraceStep key={i} event={ev} />
      ))}
    </div>
  );
}
