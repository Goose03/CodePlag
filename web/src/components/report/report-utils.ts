// Pure helpers shared by the report tabs (kept out of components so they are easy to test).
import { pairFeatures, scoreCombiner, verdictFor } from "@/lib/combiner";
import type { CombinerParams, PairState, Verdict } from "@/lib/types";

export const VERDICT_LABEL: Record<Verdict, string> = {
  likely_plagiarism: "Probable plagio",
  unlikely_plagiarism: "Sin indicios",
};

export const verdictLabel = (v: Verdict | null) => (v ? VERDICT_LABEL[v] : "calculando…");

/** Score used to order pairs: final score when available, else the provisional AST score. */
export const sortScore = (p: PairState) => p.score ?? p.astScore;

/**
 * Client-side re-score of a final pair with the job's combiner (for the what-if slider).
 * Returns the official score when the pair was scored by another combiner version, when a
 * feature is missing, or when the server applied a rule the client cannot reproduce (e.g.
 * an empty file forces 0); in that last case `mismatch` is true.
 */
export function rescore(p: PairState, c: CombinerParams | null): { score: number; mismatch: boolean } | null {
  if (p.stage !== "final" || p.score === null) return null;
  if (!c || p.combinerVersion !== c.version) return { score: p.score, mismatch: false };
  let s: number;
  try {
    s = scoreCombiner(c, pairFeatures(p.astFeatures, p.neuralScore));
  } catch {
    return { score: p.score, mismatch: false };
  }
  if (Math.abs(s - p.score) > 1e-6) return { score: p.score, mismatch: true };
  return { score: s, mismatch: false };
}

export interface WhatIf {
  threshold: number;
  /** pairKey -> verdict under the what-if threshold (final pairs only). */
  verdicts: Map<string, Verdict>;
  changed: number;
  flagged: number;
  mismatches: number;
}

/** Verdicts with a hypothetical threshold. Never touches the official verdict. O(P). */
export function computeWhatIf(pairs: PairState[], c: CombinerParams | null, threshold: number): WhatIf {
  const verdicts = new Map<string, Verdict>();
  let changed = 0;
  let flagged = 0;
  let mismatches = 0;
  for (const p of pairs) {
    const r = rescore(p, c);
    if (!r) continue;
    if (r.mismatch) mismatches += 1;
    const v = verdictFor(r.score, threshold);
    verdicts.set(p.pairKey, v);
    if (v === "likely_plagiarism") flagged += 1;
    if (v !== p.verdict) changed += 1;
  }
  return { threshold, verdicts, changed, flagged, mismatches };
}

/** Fixed-width histogram over [0, 1]. Returns bin centers and counts. */
export function histogram(values: number[], bins = 20): { x: number[]; y: number[]; width: number } {
  const width = 1 / bins;
  const y = new Array<number>(bins).fill(0);
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const i = Math.min(bins - 1, Math.max(0, Math.floor(v / width)));
    y[i] += 1;
  }
  return { x: y.map((_, i) => (i + 0.5) * width), y, width };
}

/**
 * File x file matrix of the max final score between each pair of file names.
 * With more than `limit` files keeps the `limit` files with the highest max score.
 */
export function fileMatrix(
  pairs: PairState[],
  fileNames: string[],
  limit = 60,
): { names: string[]; z: (number | null)[][]; truncated: boolean } {
  const best = new Map<string, number>();
  const cell = new Map<string, number>();
  for (const p of pairs) {
    if (p.score === null) continue;
    for (const n of [p.fileA, p.fileB]) best.set(n, Math.max(best.get(n) ?? 0, p.score));
    for (const k of [`${p.fileA}\u0000${p.fileB}`, `${p.fileB}\u0000${p.fileA}`]) {
      cell.set(k, Math.max(cell.get(k) ?? 0, p.score));
    }
  }
  let names = Array.from(new Set(fileNames.length ? fileNames : Array.from(best.keys())));
  const truncated = names.length > limit;
  if (truncated) names = [...names].sort((a, b) => (best.get(b) ?? 0) - (best.get(a) ?? 0)).slice(0, limit);
  const z = names.map((a) => names.map((b) => (a === b ? null : (cell.get(`${a}\u0000${b}`) ?? null))));
  return { names, z, truncated };
}
