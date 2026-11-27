import { cn } from "@/lib/utils";

export function Progress({
  value,
  className,
  "aria-label": ariaLabel,
}: {
  value: number;
  className?: string;
  "aria-label"?: string;
}) {
  const v = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  return (
    <div
      role="progressbar"
      aria-label={ariaLabel}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(v)}
      className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      <div className="h-full bg-primary transition-[width] duration-300" style={{ width: `${v}%` }} />
    </div>
  );
}
