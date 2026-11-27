import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { reportKey } from "@/hooks/useJobReport";
import { combineMany, initialState } from "@/lib/report-reducer";
import type { ReportState } from "@/lib/types";
import { GOLDEN_EVENTS, SAMPLE_EVENTS } from "@/test/fixtures";
import { DISCLAIMER, ReportViewer } from "./ReportViewer";

function renderWith(state: ReportState) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  // Seed the cache: with staleTime Infinity and a completed status the hook neither fetches
  // nor opens an EventSource.
  qc.setQueryData(reportKey(state.jobId), state);
  return render(
    <QueryClientProvider client={qc}>
      <ReportViewer jobId={state.jobId} />
    </QueryClientProvider>,
  );
}

describe("ReportViewer (smoke)", () => {
  const events = GOLDEN_EVENTS ?? SAMPLE_EVENTS;
  const state = combineMany(initialState("job-test"), events);

  it("renderiza un reporte completado con resumen, pestañas y disclaimer", () => {
    renderWith(state);
    expect(screen.getByText(DISCLAIMER)).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Resumen" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: `Tabla (${Object.keys(state.pairs).length})` })).toBeTruthy();
    expect((screen.getByRole("tab", { name: "PDF" }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByText("Pares marcados")).toBeTruthy();
    // Completed: no progress bar for the analysis.
    expect(screen.queryByRole("progressbar", { name: "Progreso del análisis" })).toBeNull();
  });

  it("muestra la barra de progreso y deshabilita PDF mientras corre", () => {
    const running = combineMany(initialState("job-run"), events.slice(0, Math.min(8, events.length - 1)));
    renderWith(running);
    expect(screen.getByRole("progressbar", { name: "Progreso del análisis" })).toBeTruthy();
    expect((screen.getByRole("tab", { name: "PDF" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("muestra la alerta de fallo", () => {
    const failed = combineMany(initialState("job-fail"), [
      ...events.slice(0, 3),
      { type: "job.failed", seq: 100, error: "boom", retryable: false },
    ]);
    renderWith(failed);
    expect(screen.getByText("El análisis falló")).toBeTruthy();
    expect(screen.getByText("boom")).toBeTruthy();
  });

  it("abre el detalle del par al hacer clic en el top 5", () => {
    renderWith(state);
    const top = Object.values(state.pairs).sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0];
    fireEvent.click(screen.getAllByText(`${top.fileA} ↔ ${top.fileB}`)[0]);
    expect(screen.getByRole("dialog")).toBeTruthy();
  });
});
