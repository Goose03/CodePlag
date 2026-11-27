import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Variant = "default" | "destructive" | "warning";
const variants: Record<Variant, string> = {
  default: "border-border bg-background",
  destructive: "border-destructive/50 bg-destructive/10 text-destructive",
  warning: "border-warning/60 bg-warning/15 text-warning-foreground",
};

export function Alert({
  className,
  variant = "default",
  ...p
}: HTMLAttributes<HTMLDivElement> & { variant?: Variant }) {
  return <div role="alert" className={cn("rounded-lg border px-4 py-3 text-sm", variants[variant], className)} {...p} />;
}
export function AlertTitle({ className, ...p }: HTMLAttributes<HTMLHeadingElement>) {
  return <h5 className={cn("mb-1 font-medium", className)} {...p} />;
}
export function AlertDescription({ className, ...p }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("text-sm", className)} {...p} />;
}
