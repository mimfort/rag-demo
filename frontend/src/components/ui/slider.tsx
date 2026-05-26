"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Простой нативный range-slider с tailwind-стилизацией.
 * Не используем @radix-ui/react-slider чтобы не плодить зависимости —
 * нативный input[type=range] нам полностью подходит.
 */
type SliderProps = Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  "type" | "onChange"
> & {
  value: number;
  onChange: (value: number) => void;
};

export const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, value, onChange, min = 0, max = 1, step = 0.01, ...props }, ref) => {
    return (
      <input
        ref={ref}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={cn(
          "w-full h-1 bg-muted rounded-full appearance-none cursor-pointer accent-primary",
          "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background",
          className,
        )}
        {...props}
      />
    );
  },
);
Slider.displayName = "Slider";
