"use client";
import { useMemo, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Sheet } from "@/components/ui/sheet";
import { apiUrls, fetchText } from "@/lib/api";
import type { FileRef, MatchRegion, PairState } from "@/lib/types";
import { fmtScore } from "@/lib/utils";
import { verdictLabel } from "./report-utils";

// One color per match region, used on both sides so A and B regions can be paired visually.
const REGION_COLORS = [
  "rgba(239, 68, 68, 0.22)",
  "rgba(59, 130, 246, 0.22)",
  "rgba(34, 197, 94, 0.22)",
  "rgba(234, 179, 8, 0.28)",
  "rgba(168, 85, 247, 0.22)",
  "rgba(236, 72, 153, 0.22)",
  "rgba(20, 184, 166, 0.24)",
  "rgba(249, 115, 22, 0.24)",
];
const regionColor = (i: number) => REGION_COLORS[i % REGION_COLORS.length];

/** line number (1-based) -> index of the first region covering it. */
export function lineRegions(matches: MatchRegion[], side: "linesA" | "linesB"): Map<number, number> {
  const out = new Map<number, number>();
  matches.forEach((m, i) => {
    const [from, to] = m[side];
    for (let l = Math.max(1, from); l <= to; l++) if (!out.has(l)) out.set(l, i);
  });
  return out;
}

function useSource(jobId: string, file: FileRef | undefined) {
  return useQuery({
    queryKey: ["file", jobId, file?.fileId],
    queryFn: ({ signal }) => fetchText(apiUrls.file(jobId, (file as FileRef).fileId), signal),
    enabled: file !== undefined,
    staleTime: Infinity,
  });
}

function CodePane({
  title,
  file,
  jobId,
  regions,
  paneRef,
  side,
}: {
  title: string;
  file: FileRef | undefined;
  jobId: string;
  regions: Map<number, number>;
  paneRef: React.RefObject<HTMLDivElement>;
  side: "A" | "B";
}) {
  const src = useSource(jobId, file);
  const lines = useMemo(() => (src.data ?? "").replace(/\r\n?/g, "\n").split("\n"), [src.data]);
  return (
    <div className="flex min-w-0 flex-col rounded-md border border-border">
      <div className="truncate border-b border-border bg-muted px-3 py-1.5 text-sm font-medium" title={title}>
        {title}
        {file && <span className="ml-2 text-xs font-normal text-muted-foreground">{file.language}</span>}
      </div>
      <div ref={paneRef} className="relative h-[60vh] overflow-auto">
        {!file ? (
          <p className="p-3 text-sm text-muted-foreground">Archivo no encontrado en el reporte.</p>
        ) : src.isLoading ? (
          <p className="p-3 text-sm text-muted-foreground">Cargando código…</p>
        ) : src.isError ? (
          <p className="p-3 text-sm text-destructive">No se pudo cargar: {(src.error as Error).message}</p>
        ) : (
          <pre className="min-w-max py-1 font-mono text-xs leading-5">
            {lines.map((text, i) => {
              const n = i + 1;
              const r = regions.get(n);
              return (
                <div
                  key={n}
                  data-line={`${side}-${n}`}
                  className="flex"
                  style={r !== undefined ? { backgroundColor: regionColor(r) } : undefined}
                >
                  <span className="sticky left-0 w-12 shrink-0 select-none bg-background/80 pr-3 text-right text-muted-foreground">
                    {n}
                  </span>
                  <code className="pr-4">{text || " "}</code>
                </div>
              );
            })}
          </pre>
        )}
      </div>
    </div>
  );
}

function scrollToLine(pane: HTMLDivElement | null, side: "A" | "B", line: number) {
  const el = pane?.querySelector<HTMLElement>(`[data-line="${side}-${line}"]`);
  if (pane && el) pane.scrollTo({ top: Math.max(0, el.offsetTop - 40), behavior: "smooth" });
}

export function PairDetailSheet({
  jobId,
  pair,
  files,
  onClose,
}: {
  jobId: string;
  pair: PairState | null;
  files: FileRef[];
  onClose: () => void;
}) {
  const paneA = useRef<HTMLDivElement>(null);
  const paneB = useRef<HTMLDivElement>(null);
  const matches = useMemo(() => pair?.matches ?? [], [pair]);
  const regionsA = useMemo(() => lineRegions(matches, "linesA"), [matches]);
  const regionsB = useMemo(() => lineRegions(matches, "linesB"), [matches]);

  if (!pair) return null;
  // Map file names from the event to fileId (sha256) through report.files.
  const fileA = files.find((f) => f.name === pair.fileA);
  const fileB = files.find((f) => f.name === pair.fileB);
  const feats = Object.entries(pair.astFeatures);

  return (
    <Sheet
      open
      onClose={onClose}
      title={`${pair.fileA} ↔ ${pair.fileB}`}
      description={
        <span className="flex flex-wrap items-center gap-2">
          <span>Score {pair.stage === "final" ? fmtScore(pair.score, 4) : "calculando…"}</span>
          <Badge variant={pair.verdict === "likely_plagiarism" ? "destructive" : "secondary"}>
            {verdictLabel(pair.verdict)}
          </Badge>
          <span>AST {fmtScore(pair.astScore)}</span>
          <span>Neural {fmtScore(pair.neuralScore)}</span>
          {feats.map(([k, v]) => (
            <span key={k}>
              {k} {fmtScore(v)}
            </span>
          ))}
          {pair.detector && <span>· {pair.detector}</span>}
        </span>
      }
    >
      <div className="space-y-3">
        {pair.warnings.length > 0 && (
          <ul className="list-inside list-disc text-sm text-amber-700 dark:text-amber-400">
            {pair.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
        {matches.length > 0 ? (
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="self-center text-muted-foreground">Regiones coincidentes:</span>
            {matches.map((m, i) => (
              <button
                key={i}
                type="button"
                className="rounded border border-border px-2 py-1 hover:ring-2 hover:ring-primary"
                style={{ backgroundColor: regionColor(i) }}
                onClick={() => {
                  scrollToLine(paneA.current, "A", m.linesA[0]);
                  scrollToLine(paneB.current, "B", m.linesB[0]);
                }}
              >
                A {m.linesA[0]}–{m.linesA[1]} ↔ B {m.linesB[0]}–{m.linesB[1]} · {m.node} ({m.size} nodos)
              </button>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {pair.stage === "final" ? "Sin regiones coincidentes reportadas." : "Las regiones aparecen al terminar el par."}
          </p>
        )}
        <div className="grid gap-3 md:grid-cols-2">
          <CodePane title={pair.fileA} file={fileA} jobId={jobId} regions={regionsA} paneRef={paneA} side="A" />
          <CodePane title={pair.fileB} file={fileB} jobId={jobId} regions={regionsB} paneRef={paneB} side="B" />
        </div>
      </div>
    </Sheet>
  );
}
