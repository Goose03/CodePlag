// TypeScript port of codeplag.similarity.combiner.Combiner.score.
// Used by the "what-if" threshold slider to re-score pairs in the browser. Parity with the
// Python implementation is checked against web/shared/combiner_golden.json.
import type { CombinerParams, Verdict } from "./types";

/** Numerically stable logistic function: never returns NaN for large |z|. */
export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/**
 * Final score in [0, 1] for one feature dictionary.
 * O(k) with k = number of features (at most 4).
 * @throws Error if a required feature is missing or not a finite number.
 */
export function scoreCombiner(
  c: Pick<CombinerParams, "method" | "features" | "coef" | "intercept">,
  features: Record<string, number>,
): number {
  const x = c.features.map((name) => {
    const v = features[name];
    if (v === undefined) throw new Error(`Missing feature "${name}"`);
    if (!Number.isFinite(v)) throw new Error(`Feature "${name}" is not finite: ${v}`);
    return v;
  });
  if (c.method === "weighted") {
    const total = c.coef.reduce((a, b) => a + b, 0) || 1;
    const s = c.coef.reduce((acc, w, i) => acc + w * x[i], 0) / total;
    return Math.min(1, Math.max(0, s));
  }
  const z = c.coef.reduce((acc, w, i) => acc + w * x[i], c.intercept);
  return sigmoid(z);
}

export const verdictFor = (score: number, threshold: number): Verdict =>
  score >= threshold ? "likely_plagiarism" : "unlikely_plagiarism";

/** Features of a pair as the combiner expects them (AST features + "neural" when present). */
export function pairFeatures(
  astFeatures: Record<string, number>,
  neuralScore: number | null,
): Record<string, number> {
  return neuralScore === null ? { ...astFeatures } : { ...astFeatures, neural: neuralScore };
}
