"use client";
import { createContext, useContext, useState, type KeyboardEvent, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Ctx {
  value: string;
  setValue: (v: string) => void;
}
const TabsCtx = createContext<Ctx | null>(null);
const useTabs = () => {
  const c = useContext(TabsCtx);
  if (!c) throw new Error("Tabs.* must be used inside <Tabs>");
  return c;
};

export function Tabs({
  defaultValue,
  value: controlled,
  onValueChange,
  className,
  children,
}: {
  defaultValue: string;
  value?: string;
  onValueChange?: (v: string) => void;
  className?: string;
  children: ReactNode;
}) {
  const [inner, setInner] = useState(defaultValue);
  const value = controlled ?? inner;
  const setValue = (v: string) => {
    setInner(v);
    onValueChange?.(v);
  };
  return (
    <TabsCtx.Provider value={{ value, setValue }}>
      <div className={cn("space-y-3", className)}>{children}</div>
    </TabsCtx.Provider>
  );
}

export function TabsList({ className, children }: { className?: string; children: ReactNode }) {
  // Arrow keys move between enabled triggers (WAI-ARIA tabs pattern).
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const tabs = Array.from(e.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]:not(:disabled)'));
    const i = tabs.indexOf(document.activeElement as HTMLButtonElement);
    if (i < 0) return;
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next?.focus();
    next?.click();
  };
  return (
    <div
      role="tablist"
      onKeyDown={onKeyDown}
      className={cn("inline-flex flex-wrap items-center gap-1 rounded-lg bg-muted p-1", className)}
    >
      {children}
    </div>
  );
}

export function TabsTrigger({
  value,
  disabled,
  className,
  children,
}: {
  value: string;
  disabled?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const t = useTabs();
  const active = t.value === value;
  return (
    <button
      type="button"
      role="tab"
      id={`tab-${value}`}
      aria-selected={active}
      aria-controls={`panel-${value}`}
      tabIndex={active ? 0 : -1}
      disabled={disabled}
      onClick={() => t.setValue(value)}
      className={cn(
        "rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        active ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground",
        className,
      )}
    >
      {children}
    </button>
  );
}

/** Only the active panel is mounted (charts and grids size themselves on mount). */
export function TabsContent({ value, className, children }: { value: string; className?: string; children: ReactNode }) {
  const t = useTabs();
  if (t.value !== value) return null;
  return (
    <div role="tabpanel" id={`panel-${value}`} aria-labelledby={`tab-${value}`} className={className}>
      {children}
    </div>
  );
}
