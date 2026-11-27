import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { readShared } from "@/test/fixtures";
import { pairFeatures, scoreCombiner, sigmoid, verdictFor } from "./combiner";
import type { CombinerParams } from "./types";

interface GoldenCase {
  combiner: CombinerParams;
  features: Record<string, number>;
  expected: number;
}
const golden = readShared<GoldenCase[]>("combiner_golden.json") ?? [];

const logistic: CombinerParams = {
  name: "combined",
  method: "logistic",
  features: ["ted", "subtree", "ngram", "neural"],
  coef: [0.228, 3.596, 3.357, 5.153],
  intercept: -4.519,
  threshold: 0.1766,
  version: "t",
};
const weighted: CombinerParams = {
  name: "weighted",
  method: "weighted",
  features: ["ast", "neural"],
  coef: [0.5, 0.5],
  intercept: 0,
  threshold: 0.5,
  version: "t",
};

describe("scoreCombiner: paridad con Python (combiner_golden.json)", () => {
  it.runIf(golden.length > 0)(`hay casos golden (${golden.length})`, () => {
    expect(golden.length).toBeGreaterThan(0);
  });
  it.each(golden.map((c, i) => [i, c] as const))("caso #%i", (_i, c) => {
    expect(scoreCombiner(c.combiner, c.features)).toBeCloseTo(c.expected, 12);
  });
});

describe("scoreCombiner", () => {
  it("logístico: coincide con la fórmula", () => {
    const f = { ted: 0.5, subtree: 0.4, ngram: 0.3, neural: 0.9 };
    const z = -4.519 + 0.228 * 0.5 + 3.596 * 0.4 + 3.357 * 0.3 + 5.153 * 0.9;
    expect(scoreCombiner(logistic, f)).toBeCloseTo(1 / (1 + Math.exp(-z)), 14);
  });

  it("lanza si falta una feature", () => {
    expect(() => scoreCombiner(logistic, { ted: 1, subtree: 1, ngram: 1 })).toThrow(/neural/);
  });

  it("lanza si una feature no es finita", () => {
    expect(() => scoreCombiner(logistic, { ted: NaN, subtree: 1, ngram: 1, neural: 1 })).toThrow(/ted/);
    expect(() => scoreCombiner(logistic, { ted: 1, subtree: Infinity, ngram: 1, neural: 1 })).toThrow(/subtree/);
  });

  it("ignora features extra", () => {
    const f = { ted: 0.5, subtree: 0.4, ngram: 0.3, neural: 0.9 };
    expect(scoreCombiner(logistic, { ...f, extra: 123 })).toBe(scoreCombiner(logistic, f));
  });

  it("weighted: promedio ponderado recortado a [0, 1]", () => {
    expect(scoreCombiner(weighted, { ast: 0.2, neural: 0.6 })).toBeCloseTo(0.4, 14);
    expect(scoreCombiner({ ...weighted, coef: [1, 1] }, { ast: 3, neural: 3 })).toBe(1);
    expect(scoreCombiner(weighted, { ast: -2, neural: -2 })).toBe(0);
  });

  it("weighted: suma de coeficientes 0 usa total = 1 (como Python `or 1`)", () => {
    expect(scoreCombiner({ ...weighted, coef: [1, -1] }, { ast: 0.8, neural: 0.3 })).toBeCloseTo(0.5, 14);
  });

  it("propiedad: el score siempre está en [0, 1] para features en [0, 1]", () => {
    const unit = fc.double({ min: 0, max: 1, noNaN: true });
    fc.assert(
      fc.property(
        fc.record({ ted: unit, subtree: unit, ngram: unit, neural: unit }),
        fc.array(fc.double({ min: -50, max: 50, noNaN: true }), { minLength: 4, maxLength: 4 }),
        fc.double({ min: -50, max: 50, noNaN: true }),
        (f, coef, intercept) => {
          for (const method of ["logistic", "weighted"] as const) {
            const s = scoreCombiner({ ...logistic, method, coef, intercept }, f);
            expect(s).toBeGreaterThanOrEqual(0);
            expect(s).toBeLessThanOrEqual(1);
            expect(Number.isNaN(s)).toBe(false);
          }
        },
      ),
    );
  });

  it("propiedad: logístico monótono en una feature con coeficiente positivo", () => {
    const unit = fc.double({ min: 0, max: 1, noNaN: true });
    fc.assert(
      fc.property(unit, unit, unit, unit, unit, (a, b, ted, subtree, ngram) => {
        const [lo, hi] = a <= b ? [a, b] : [b, a];
        const base = { ted, subtree, ngram };
        expect(scoreCombiner(logistic, { ...base, neural: hi })).toBeGreaterThanOrEqual(
          scoreCombiner(logistic, { ...base, neural: lo }),
        );
      }),
    );
  });
});

describe("sigmoid", () => {
  it("estable en extremos", () => {
    expect(sigmoid(-1e4)).toBe(0);
    expect(sigmoid(1e4)).toBe(1);
    expect(Number.isNaN(sigmoid(-800))).toBe(false);
    expect(sigmoid(0)).toBe(0.5);
  });
  it("simétrica: sigmoid(-z) = 1 - sigmoid(z)", () => {
    fc.assert(
      fc.property(fc.double({ min: -30, max: 30, noNaN: true }), (z) => {
        expect(sigmoid(-z)).toBeCloseTo(1 - sigmoid(z), 12);
      }),
    );
  });
});

describe("verdictFor / pairFeatures", () => {
  it("score igual al umbral cuenta como probable plagio (>=)", () => {
    expect(verdictFor(0.5, 0.5)).toBe("likely_plagiarism");
    expect(verdictFor(0.4999, 0.5)).toBe("unlikely_plagiarism");
  });
  it("añade neural sólo si existe", () => {
    expect(pairFeatures({ ted: 1 }, null)).toEqual({ ted: 1 });
    expect(pairFeatures({ ted: 1 }, 0.3)).toEqual({ ted: 1, neural: 0.3 });
  });
  it.runIf(golden.length > 0)("el veredicto golden es coherente con el umbral de cada combiner", () => {
    for (const c of golden) {
      const s = scoreCombiner(c.combiner, c.features);
      expect(verdictFor(s, c.combiner.threshold)).toBe(verdictFor(c.expected, c.combiner.threshold));
    }
  });
});
