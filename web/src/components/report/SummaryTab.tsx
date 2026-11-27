"use client";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { PairState, ReportState } from "@/lib/types";
import { fmtScore } from "@/lib/utils";
import { sortScore, verdictLabel } from "./report-utils";

function Stat({ title, value, hint }: { title: string; value: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold tabular-nums">{value}</div>
        {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
      </CardContent>
    </Card>
  );
}

export function SummaryTab({
  report,
  pairs,
  onSelect,
}: {
  report: ReportState;
  pairs: PairState[];
  onSelect: (pairKey: string) => void;
}) {
  const finals = pairs.filter((p) => p.stage === "final").length;
  const top = pairs.slice(0, 5);
  const c = report.combiner;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <Stat title="Archivos" value={report.files.length} />
        <Stat
          title="Pares candidatos"
          value={`${report.candidatePairs} / ${report.totalPairs}`}
          hint={`Prefiltro: ${report.prefilter ?? "—"}`}
        />
        <Stat title="Pares evaluados" value={`${finals} / ${report.total || report.candidatePairs}`} />
        <Stat
          title="Pares marcados"
          value={<span className={report.flagged > 0 ? "text-destructive" : undefined}>{report.flagged}</span>}
          hint="Probable plagio (umbral calibrado)"
        />
        <Stat
          title="Combiner"
          value={<span className="text-base">{c?.name ?? "—"}</span>}
          hint={c ? `Umbral ${c.threshold.toFixed(4)} · v${c.version}` : undefined}
        />
        <Stat
          title="Tiempo"
          value={report.elapsed !== null ? `${report.elapsed.toFixed(1)} s` : "—"}
          hint={report.modelVersion ? `Modelo: ${report.modelVersion}` : "Sin modelo neuronal"}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pares con mayor score</CardTitle>
        </CardHeader>
        <CardContent>
          {top.length === 0 ? (
            <p className="text-sm text-muted-foreground">Aún no hay pares evaluados.</p>
          ) : (
            <ol className="space-y-2">
              {top.map((p) => {
                const s = sortScore(p);
                return (
                  <li key={p.pairKey}>
                    <button
                      type="button"
                      onClick={() => onSelect(p.pairKey)}
                      className="w-full rounded-md p-2 text-left hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                        <span className="min-w-0 truncate font-medium">
                          {p.fileA} ↔ {p.fileB}
                        </span>
                        <span className="flex items-center gap-2">
                          <span className="tabular-nums">{p.stage === "final" ? fmtScore(p.score) : "calculando…"}</span>
                          <Badge
                            variant={
                              p.verdict === "likely_plagiarism"
                                ? "destructive"
                                : p.verdict === null
                                  ? "outline"
                                  : "secondary"
                            }
                          >
                            {verdictLabel(p.verdict)}
                          </Badge>
                        </span>
                      </div>
                      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                        <div
                          className={
                            p.verdict === "likely_plagiarism" ? "h-full bg-destructive" : "h-full bg-primary/70"
                          }
                          style={{ width: `${Math.round(Math.max(0, Math.min(1, s)) * 100)}%` }}
                        />
                      </div>
                    </button>
                  </li>
                );
              })}
            </ol>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
