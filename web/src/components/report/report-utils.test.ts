import { describe, expect, it } from "vitest";
import { combineMany, initialState } from "@/lib/report-reducer";
import { EVENT_DATASETS, SAMPLE_EVENTS } from "@/test/fixtures";
import { computeWhatIf, fileMatrix, histogram, rescore } from "./report-utils";
import { lineRegions } from "./PairDetailSheet";

describe.each(EVENT_DATASETS)("what-if sobre %s", (_name, events) => {
  const state = combineMany(initialState("j"), events);
  const pairs = Object.values(state.pairs);

  it("con el umbral calibrado no cambia ningún veredicto", () => {
    const wi = computeWhatIf(pairs, state.combiner, state.combiner?.threshold ?? 0.5);
    expect(wi.changed).toBe(0);
    expect(wi.flagged).toBe(state.flagged);
  });

  // Only meaningful for real data: the hand-written sample uses made-up scores.
  it.runIf(events !== SAMPLE_EVENTS)("el re-score en el cliente coincide con el score del servidor", () => {
    const current = pairs.filter((p) => p.combinerVersion === state.combiner?.version);
    for (const p of current) {
      const r = rescore(p, state.combiner);
      expect(r?.mismatch).toBe(false);
      expect(r?.score).toBeCloseTo(p.score as number, 9);
    }
  });

  it("umbral 0 marca todo y umbral > 1 no marca nada", () => {
    expect(computeWhatIf(pairs, state.combiner, 0).flagged).toBe(pairs.length);
    expect(computeWhatIf(pairs, state.combiner, 1.01).flagged).toBe(0);
  });

  it("nunca modifica los pares originales", () => {
    const before = JSON.stringify(pairs);
    computeWhatIf(pairs, state.combiner, 0.9);
    expect(JSON.stringify(pairs)).toBe(before);
  });
});

describe("histogram", () => {
  it("cuenta cada valor en su bin (1.0 cae en el último)", () => {
    const h = histogram([0, 0.04, 0.5, 1], 20);
    expect(h.y.reduce((a, b) => a + b, 0)).toBe(4);
    expect(h.y[0]).toBe(2);
    expect(h.y[10]).toBe(1);
    expect(h.y[19]).toBe(1);
  });
});

describe("fileMatrix", () => {
  const p = (fileA: string, fileB: string, score: number | null) =>
    ({ fileA, fileB, score }) as Parameters<typeof fileMatrix>[0][number];

  it("es simétrica con diagonal vacía", () => {
    const m = fileMatrix([p("a", "b", 0.9), p("a", "c", 0.1)], ["a", "b", "c"]);
    expect(m.z[0][1]).toBe(0.9);
    expect(m.z[1][0]).toBe(0.9);
    expect(m.z[0][0]).toBeNull();
    expect(m.z[1][2]).toBeNull();
  });

  it("con más archivos que el límite conserva los de mayor score", () => {
    const m = fileMatrix([p("a", "b", 0.9), p("c", "d", 0.2)], ["a", "b", "c", "d"], 2);
    expect(m.truncated).toBe(true);
    expect(m.names.sort()).toEqual(["a", "b"]);
  });
});

describe("lineRegions", () => {
  it("asigna a cada línea la primera región que la cubre", () => {
    const r = lineRegions(
      [
        { linesA: [1, 3], linesB: [5, 7], size: 1, node: "x" },
        { linesA: [3, 4], linesB: [1, 1], size: 1, node: "y" },
      ],
      "linesA",
    );
    expect([1, 2, 3, 4].map((l) => r.get(l))).toEqual([0, 0, 0, 1]);
    expect(r.has(5)).toBe(false);
  });
});
