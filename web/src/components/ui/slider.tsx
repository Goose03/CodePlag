import { cn } from "@/lib/utils";

export function Slider({
  value,
  onValueChange,
  min = 0,
  max = 1,
  step = 0.01,
  className,
  "aria-label": ariaLabel,
}: {
  value: number;
  onValueChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  "aria-label"?: string;
}) {
  return (
    <input
      type="range"
      aria-label={ariaLabel}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onValueChange(Number(e.target.value))}
      className={cn("h-2 w-full cursor-pointer accent-[hsl(var(--primary))]", className)}
    />
  );
}
