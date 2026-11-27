"use client";
import { useMemo, useRef, useState } from "react";
import {
  AllCommunityModule,
  ModuleRegistry,
  type ColDef,
  type GetRowIdParams,
  type RowClassRules,
  type RowClickedEvent,
  type ValueFormatterParams,
} from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PairState, Verdict } from "@/lib/types";
import { VERDICT_LABEL } from "./report-utils";

ModuleRegistry.registerModules([AllCommunityModule]);

interface Row {
  pairKey: string;
  stage: "ast" | "final";
  fileA: string;
  fileB: string;
  score: number | null;
  verdict: string;
  whatIfVerdict: string;
  astScore: number;
  neuralScore: number | null;
  ted: number | null;
  subtree: number | null;
  ngram: number | null;
  detector: string;
  warnings: string;
}

const PENDING = "calculando…";
const num =
  (digits = 3) =>
  (p: ValueFormatterParams<Row>) => {
    if (p.data?.stage === "ast" && (p.colDef.field === "score" || p.colDef.field === "neuralScore")) return PENDING;
    return typeof p.value === "number" ? p.value.toFixed(digits) : "—";
  };
const feat = (p: PairState, k: string) => (typeof p.astFeatures[k] === "number" ? p.astFeatures[k] : null);

export function PairsTable({
  pairs,
  onSelect,
  whatIf,
  jobId,
}: {
  pairs: PairState[];
  onSelect: (pairKey: string) => void;
  /** Hypothetical verdicts from the what-if slider (null = slider at the calibrated threshold). */
  whatIf?: { threshold: number; verdicts: Map<string, Verdict> } | null;
  jobId?: string;
}) {
  const grid = useRef<AgGridReact<Row>>(null);
  const [quick, setQuick] = useState("");

  const rows = useMemo<Row[]>(
    () =>
      pairs.map((p) => {
        const wv = whatIf?.verdicts.get(p.pairKey);
        return {
          pairKey: p.pairKey,
          stage: p.stage,
          fileA: p.fileA,
          fileB: p.fileB,
          score: p.score,
          verdict: p.verdict ? VERDICT_LABEL[p.verdict] : PENDING,
          whatIfVerdict: wv ? VERDICT_LABEL[wv] : p.stage === "ast" ? PENDING : "—",
          astScore: p.astScore,
          neuralScore: p.neuralScore,
          ted: feat(p, "ted"),
          subtree: feat(p, "subtree"),
          ngram: feat(p, "ngram"),
          detector: p.detector ?? "",
          warnings: p.warnings.join(" | "),
        };
      }),
    [pairs, whatIf],
  );

  const columns = useMemo<ColDef<Row>[]>(() => {
    const numCol = (field: keyof Row, headerName: string, width = 110): ColDef<Row> => ({
      field,
      headerName,
      width,
      filter: "agNumberColumnFilter",
      valueFormatter: num(),
      type: "rightAligned",
    });
    const cols: ColDef<Row>[] = [
      { field: "fileA", headerName: "Archivo A", filter: "agTextColumnFilter", minWidth: 150, flex: 1 },
      { field: "fileB", headerName: "Archivo B", filter: "agTextColumnFilter", minWidth: 150, flex: 1 },
      { ...numCol("score", "Score"), sort: "desc" },
      {
        field: "verdict",
        headerName: "Veredicto",
        filter: "agTextColumnFilter",
        width: 150,
        cellClassRules: { "text-red-600 font-medium": (p) => p.value === VERDICT_LABEL.likely_plagiarism },
      },
    ];
    if (whatIf) {
      cols.push({
        field: "whatIfVerdict",
        headerName: `Veredicto con umbral ${whatIf.threshold.toFixed(2)}`,
        filter: "agTextColumnFilter",
        width: 210,
      });
    }
    cols.push(
      numCol("astScore", "AST"),
      numCol("neuralScore", "Neural"),
      numCol("ted", "ted", 95),
      numCol("subtree", "subtree", 100),
      numCol("ngram", "ngram", 95),
      { field: "detector", headerName: "Detector", filter: "agTextColumnFilter", width: 130 },
      { field: "warnings", headerName: "Avisos", filter: "agTextColumnFilter", minWidth: 160, flex: 1, tooltipField: "warnings" },
    );
    return cols;
  }, [whatIf]);

  const rowClassRules = useMemo<RowClassRules<Row>>(() => ({ "row-pending": (p) => p.data?.stage === "ast" }), []);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={quick}
          onChange={(e) => setQuick(e.target.value)}
          placeholder="Buscar en la tabla…"
          aria-label="Búsqueda rápida"
          className="h-9 w-64 rounded-md border border-border bg-background px-3 text-sm"
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            grid.current?.api.exportDataAsCsv({ fileName: `codeplag-${jobId ?? "pares"}.csv`, columnKeys: undefined })
          }
        >
          <Download className="h-4 w-4" /> Exportar CSV
        </Button>
        <span className="text-xs text-muted-foreground">Clic en una fila para ver el código lado a lado.</span>
      </div>
      <div style={{ height: 560 }}>
        <AgGridReact<Row>
          ref={grid}
          rowData={rows}
          columnDefs={columns}
          defaultColDef={{ sortable: true, resizable: true, floatingFilter: true }}
          getRowId={(p: GetRowIdParams<Row>) => p.data.pairKey}
          rowClassRules={rowClassRules}
          quickFilterText={quick}
          onRowClicked={(e: RowClickedEvent<Row>) => e.data && onSelect(e.data.pairKey)}
          animateRows={false}
          overlayNoRowsTemplate="Aún no hay pares."
          localeText={{ noRowsToShow: "Aún no hay pares." }}
        />
      </div>
    </div>
  );
}
