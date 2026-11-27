"use client";
import { useMemo } from "react";
import dynamic from "next/dynamic";
import type { Data, Layout, PlotMouseEvent, Shape } from "plotly.js";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import type { PairState, ReportState, Verdict } from "@/lib/types";
import { computeWhatIf, fileMatrix, histogram, VERDICT_LABEL } from "./report-utils";

// Plotly touches `window` on import: load it only in the browser. The cartesian bundle is the
// smallest one that includes both `histogram`-style bars and `heatmap`.
const Plot = dynamic(
  async () => {
    const [{ default: Plotly }, { default: createPlotlyComponent }] = await Promise.all([
      import("plotly.js-cartesian-dist-min"),
      import("react-plotly.js/factory"),
    ]);
    return createPlotlyComponent(Plotly);
  },
  {
    ssr: false,
    loading: () => <div className="grid h-72 place-items-center text-sm text-muted-foreground">Cargando gráfico…</div>,
  },
);

const COLORS: Record<Verdict, string> = { likely_plagiarism: "#dc2626", unlikely_plagiarism: "#2563eb" };
const HEATMAP_LIMIT = 60;
const baseLayout: Partial<Layout> = {
  autosize: true,
  margin: { l: 50, r: 20, t: 10, b: 45 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { size: 12, color: "#64748b" },
};
const config = { displaylogo: false, responsive: true } as const;

const vline = (x: number, color: string, dash: "solid" | "dash"): Partial<Shape> => ({
  type: "line",
  xref: "x",
  yref: "paper",
  x0: x,
  x1: x,
  y0: 0,
  y1: 1,
  line: { color, width: 2, dash },
});

export function ChartsTab({
  report,
  pairs,
  whatIf,
  onWhatIf,
  onSelect,
}: {
  report: ReportState;
  pairs: PairState[];
  whatIf: number | null;
  onWhatIf: (t: number | null) => void;
  onSelect: (pairKey: string) => void;
}) {
  const calibrated = report.combiner?.threshold ?? null;
  const threshold = whatIf ?? calibrated ?? 0.5;
  const finals = useMemo(() => pairs.filter((p) => p.stage === "final" && p.score !== null), [pairs]);
  const wi = useMemo(() => computeWhatIf(finals, report.combiner, threshold), [finals, report.combiner, threshold]);

  const hist = useMemo(() => histogram(finals.map((p) => p.score as number)), [finals]);
  const withNeural = useMemo(() => finals.filter((p) => p.neuralScore !== null), [finals]);
  const matrix = useMemo(
    () => fileMatrix(finals, report.files.map((f) => f.name), HEATMAP_LIMIT),
    [finals, report.files],
  );

  const shapes: Partial<Shape>[] = [];
  if (calibrated !== null) shapes.push(vline(calibrated, "#dc2626", "solid"));
  if (whatIf !== null && whatIf !== calibrated) shapes.push(vline(whatIf, "#f59e0b", "dash"));

  const histData: Data[] = [
    {
      type: "bar",
      x: hist.x,
      y: hist.y,
      width: hist.x.map(() => hist.width * 0.95),
      marker: { color: "#64748b" },
      hovertemplate: "score %{x:.2f}: %{y} pares<extra></extra>",
    },
  ];

  const scatterData: Data[] = (["likely_plagiarism", "unlikely_plagiarism"] as Verdict[]).map((v) => {
    const ps = withNeural.filter((p) => p.verdict === v);
    return {
      type: "scatter",
      mode: "markers",
      name: VERDICT_LABEL[v],
      x: ps.map((p) => p.astScore),
      y: ps.map((p) => p.neuralScore as number),
      customdata: ps.map((p) => p.pairKey),
      text: ps.map((p) => `${p.fileA} ↔ ${p.fileB}<br>score ${(p.score as number).toFixed(3)}`),
      hovertemplate: "%{text}<br>AST %{x:.3f} · neural %{y:.3f}<extra></extra>",
      marker: { color: COLORS[v], size: 9, opacity: 0.8 },
    };
  });

  const onScatterClick = (e: Readonly<PlotMouseEvent>) => {
    const key = e.points[0]?.customdata;
    if (typeof key === "string") onSelect(key);
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Umbral “what-if”</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <Slider
              className="max-w-md"
              value={threshold}
              onValueChange={onWhatIf}
              aria-label="Umbral hipotético"
            />
            <span className="tabular-nums text-sm">{threshold.toFixed(2)}</span>
            {whatIf !== null && (
              <Button size="sm" variant="outline" onClick={() => onWhatIf(null)}>
                Volver al calibrado
              </Button>
            )}
          </div>
          <p className="text-sm" aria-live="polite">
            Con umbral {threshold.toFixed(2)}: <strong>{wi.flagged}</strong> pares marcados;{" "}
            <strong>{wi.changed}</strong> {wi.changed === 1 ? "par cambiaría" : "pares cambiarían"} de veredicto
            {calibrated !== null && ` respecto al umbral calibrado (${calibrated.toFixed(4)}, FPR ≤ 5 % en validación)`}.
          </p>
          <p className="text-xs text-muted-foreground">
            Simulación local: el veredicto oficial, la tabla y el PDF siempre usan el umbral calibrado.
            {wi.mismatches > 0 &&
              ` ${wi.mismatches} pares no pudieron recalcularse igual que en el servidor y conservan su score oficial.`}
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Distribución de scores</CardTitle>
          </CardHeader>
          <CardContent>
            {finals.length === 0 ? (
              <p className="text-sm text-muted-foreground">Sin pares evaluados todavía.</p>
            ) : (
              <Plot
                data={histData}
                layout={{
                  ...baseLayout,
                  height: 300,
                  shapes,
                  bargap: 0,
                  xaxis: { title: { text: "score" }, range: [0, 1] },
                  yaxis: { title: { text: "pares" } },
                }}
                config={config}
                useResizeHandler
                style={{ width: "100%" }}
              />
            )}
            <p className="mt-1 text-xs text-muted-foreground">
              Línea roja: umbral calibrado. Línea ámbar discontinua: umbral “what-if”.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>AST vs. neuronal</CardTitle>
          </CardHeader>
          <CardContent>
            {withNeural.length === 0 ? (
              <p className="text-sm text-muted-foreground">Sin scores neuronales (modelo no disponible o sin pares).</p>
            ) : (
              <Plot
                data={scatterData}
                layout={{
                  ...baseLayout,
                  height: 300,
                  xaxis: { title: { text: "score AST" }, range: [-0.02, 1.02] },
                  yaxis: { title: { text: "score neuronal" }, range: [-0.02, 1.02] },
                  legend: { orientation: "h", y: -0.25 },
                }}
                config={config}
                useResizeHandler
                style={{ width: "100%" }}
                onClick={onScatterClick}
              />
            )}
            <p className="mt-1 text-xs text-muted-foreground">Clic en un punto para ver el detalle del par.</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Mapa de calor archivo × archivo (score máximo)</CardTitle>
        </CardHeader>
        <CardContent>
          {matrix.names.length < 2 || finals.length === 0 ? (
            <p className="text-sm text-muted-foreground">Sin datos suficientes.</p>
          ) : (
            <>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    x: matrix.names,
                    y: matrix.names,
                    z: matrix.z,
                    zmin: 0,
                    zmax: 1,
                    colorscale: "Reds",
                    hoverongaps: false,
                    hovertemplate: "%{y} ↔ %{x}: %{z:.3f}<extra></extra>",
                  } as Data,
                ]}
                layout={{
                  ...baseLayout,
                  height: Math.min(900, 220 + matrix.names.length * 14),
                  margin: { l: 120, r: 20, t: 10, b: 120 },
                  yaxis: { autorange: "reversed" },
                }}
                config={config}
                useResizeHandler
                style={{ width: "100%" }}
              />
              {matrix.truncated && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Hay más de {HEATMAP_LIMIT} archivos: se muestran los {HEATMAP_LIMIT} con mayor score máximo.
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
