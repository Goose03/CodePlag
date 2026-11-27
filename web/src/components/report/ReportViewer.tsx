"use client";
import { useCallback, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useJobReport } from "@/hooks/useJobReport";
import type { PairState } from "@/lib/types";
import { ChartsTab } from "./ChartsTab";
import { LogsTab } from "./LogsTab";
import { PairDetailSheet } from "./PairDetailSheet";
import { PdfTab } from "./PdfTab";
import { computeWhatIf, sortScore } from "./report-utils";
import { SummaryTab } from "./SummaryTab";

// AG-Grid is large and only needed when the Tabla tab is opened.
const PairsTable = dynamic(() => import("./PairsTable").then((m) => m.PairsTable), {
  ssr: false,
  loading: () => <p className="text-sm text-muted-foreground">Cargando tabla…</p>,
});

export const DISCLAIMER =
  "Un score alto es evidencia para revisión humana, no prueba de plagio. Plantillas, código " +
  "base y algoritmos estándar pueden parecerse legítimamente.";

const NEURAL_MISSING = "Neural model unavailable";

export function ReportViewer({ jobId }: { jobId: string }) {
  const { data: report, error, isLoading, transport } = useJobReport(jobId);
  const [selected, setSelected] = useState<string | null>(null);
  // What-if threshold: null = calibrated. Only affects the Charts tab and an extra table column.
  const [whatIf, setWhatIf] = useState<number | null>(null);
  const close = useCallback(() => setSelected(null), []);

  const pairs = useMemo<PairState[]>(
    () => Object.values(report?.pairs ?? {}).sort((a, b) => sortScore(b) - sortScore(a)),
    [report?.pairs],
  );
  const whatIfResult = useMemo(
    () => (whatIf === null || !report ? null : computeWhatIf(pairs, report.combiner, whatIf)),
    [whatIf, pairs, report],
  );

  if (isLoading) return <p aria-busy="true">Cargando reporte…</p>;
  if (error || !report) {
    return (
      <Alert variant="destructive">
        <AlertTitle>No se pudo cargar el reporte</AlertTitle>
        <AlertDescription>{error ? (error as Error).message : "Respuesta vacía."}</AlertDescription>
      </Alert>
    );
  }

  const pct = report.total ? Math.round((report.done / report.total) * 100) : 0;
  const neuralMissing = pairs.some((p) => p.warnings.some((w) => w.startsWith(NEURAL_MISSING)));
  const running = report.status === "queued" || report.status === "running";

  return (
    <section aria-label="Reporte" className="space-y-4">
      {running && (
        <div className="space-y-1" aria-live="polite">
          <Progress value={pct} aria-label="Progreso del análisis" />
          <p className="text-sm text-muted-foreground">
            {report.status === "queued"
              ? "En cola…"
              : `${report.done}/${report.total} pares · los resultados parciales ya son visibles`}
            {transport === "polling" && " · conexión en vivo no disponible, actualizando cada 2 s"}
          </p>
        </div>
      )}
      {report.status === "failed" && (
        <Alert variant="destructive">
          <AlertTitle>El análisis falló</AlertTitle>
          <AlertDescription>{report.error ?? "Error desconocido."}</AlertDescription>
        </Alert>
      )}
      {neuralMissing && (
        <Alert variant="warning">
          <AlertDescription>Modelo neuronal no disponible: el score usa sólo el análisis AST.</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="summary">
        <TabsList>
          <TabsTrigger value="summary">Resumen</TabsTrigger>
          <TabsTrigger value="charts">Gráficos</TabsTrigger>
          <TabsTrigger value="table">Tabla ({pairs.length})</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
          <TabsTrigger value="pdf" disabled={report.status !== "completed"}>
            PDF
          </TabsTrigger>
        </TabsList>
        <TabsContent value="summary">
          <SummaryTab report={report} pairs={pairs} onSelect={setSelected} />
        </TabsContent>
        <TabsContent value="charts">
          <ChartsTab report={report} pairs={pairs} whatIf={whatIf} onWhatIf={setWhatIf} onSelect={setSelected} />
        </TabsContent>
        <TabsContent value="table">
          <PairsTable pairs={pairs} onSelect={setSelected} whatIf={whatIfResult} jobId={jobId} />
        </TabsContent>
        <TabsContent value="logs">
          <LogsTab logs={report.logs} />
        </TabsContent>
        <TabsContent value="pdf">
          <PdfTab jobId={jobId} status={report.status} />
        </TabsContent>
      </Tabs>

      <p className="text-xs text-muted-foreground">{DISCLAIMER}</p>
      <PairDetailSheet
        jobId={jobId}
        pair={selected ? (report.pairs[selected] ?? null) : null}
        files={report.files}
        onClose={close}
      />
    </section>
  );
}
