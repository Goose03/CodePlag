"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { LogLine } from "@/lib/types";
import { cn } from "@/lib/utils";

const LEVELS = ["debug", "info", "warning", "error"] as const;
const LEVEL_LABEL: Record<string, string> = { debug: "Debug", info: "Info", warning: "Aviso", error: "Error" };
const LEVEL_CLASS: Record<string, string> = {
  debug: "text-muted-foreground",
  info: "text-foreground",
  warning: "text-amber-600 dark:text-amber-400",
  error: "text-destructive",
};
const ROW_PX = 22;

export function LogsTab({ logs }: { logs: LogLine[] }) {
  const [enabled, setEnabled] = useState<Record<string, boolean>>({ debug: false, info: true, warning: true, error: true });
  const shown = useMemo(() => logs.filter((l) => enabled[l.level] ?? true), [logs, enabled]);
  const parent = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);

  const virt = useVirtualizer({
    count: shown.length,
    getScrollElement: () => parent.current,
    estimateSize: () => ROW_PX,
    overscan: 20,
  });

  // Auto-scroll only if the user was already at the bottom.
  useEffect(() => {
    if (atBottom.current && shown.length) virt.scrollToIndex(shown.length - 1, { align: "end" });
  }, [shown.length, virt]);

  return (
    <div className="space-y-2">
      <fieldset className="flex flex-wrap items-center gap-3 text-sm">
        <legend className="sr-only">Niveles visibles</legend>
        {LEVELS.map((lv) => (
          <label key={lv} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={enabled[lv] ?? true}
              onChange={(e) => setEnabled((s) => ({ ...s, [lv]: e.target.checked }))}
            />
            {LEVEL_LABEL[lv]} ({logs.filter((l) => l.level === lv).length})
          </label>
        ))}
        <span className="text-xs text-muted-foreground">
          {shown.length} de {logs.length} líneas (se guardan las últimas 2000)
        </span>
      </fieldset>
      <div
        ref={parent}
        role="log"
        aria-label="Registro del análisis"
        className="h-[480px] overflow-auto rounded-md border border-border bg-muted/30 font-mono text-xs"
        onScroll={(e) => {
          const el = e.currentTarget;
          atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < ROW_PX * 2;
        }}
      >
        {shown.length === 0 ? (
          <p className="p-3 text-muted-foreground">Sin líneas de log.</p>
        ) : (
          <div style={{ height: virt.getTotalSize(), position: "relative" }}>
            {virt.getVirtualItems().map((item) => {
              const l = shown[item.index];
              return (
                <div
                  key={l.seq}
                  className={cn("absolute left-0 right-0 flex gap-3 whitespace-pre px-3", LEVEL_CLASS[l.level])}
                  style={{ top: 0, height: item.size, transform: `translateY(${item.start}px)`, lineHeight: `${ROW_PX}px` }}
                >
                  <span className="shrink-0 text-muted-foreground">{l.ts ? l.ts.slice(11, 19) : `#${l.seq}`}</span>
                  <span className="w-14 shrink-0 uppercase">{l.level}</span>
                  <span className="truncate" title={l.message}>
                    {l.message}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
