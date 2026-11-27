// Fold job events into the report state. Same rules as web/api/reducer.py:
//  1. Idempotent: events with seq <= lastSeq are ignored (same object returned).
//  2. A jump in seq sets `gap`; the UI then refetches the server snapshot.
//  3. A pair in the "final" stage never goes back to "ast".
//  4. Two "final" results for one pair: current combiner version wins, else higher seq.
//  5. `flagged` is updated from verdict changes, never by counting events.
//  6. After job.completed / job.failed only "log" events are applied.
//  7. At most MAX_LOGS log lines are kept.
import type { JobEvent, PairScored, PairState, ReportState } from "./types";

export const MAX_LOGS = 2000;

export const initialState = (jobId: string): ReportState => ({
  jobId,
  status: "queued",
  lastSeq: 0,
  gap: false,
  files: [],
  combiner: null,
  modelVersion: null,
  prefilter: null,
  totalPairs: 0,
  candidatePairs: 0,
  done: 0,
  total: 0,
  pairs: {},
  flagged: 0,
  elapsed: null,
  error: null,
  logs: [],
});

const isFlagged = (p: PairState | undefined): number =>
  p !== undefined && p.verdict === "likely_plagiarism" ? 1 : 0;

function newWins(prev: PairState, e: PairScored, current: string | undefined): boolean {
  const prevOk = prev.combinerVersion === current;
  const nextOk = e.combinerVersion === current;
  if (prevOk !== nextOk) return nextOk;
  return e.seq > prev.updatedSeq;
}

/**
 * Apply one event. Pure and idempotent.
 * Complexity: O(1) plus an O(P) copy of `pairs` when a pair changes (P = number of pairs).
 */
export function combineResults(state: ReportState, e: JobEvent): ReportState {
  if (e.seq <= state.lastSeq) return state;
  const s: ReportState = {
    ...state,
    lastSeq: e.seq,
    gap: state.gap || e.seq > state.lastSeq + 1,
  };
  if ((state.status === "completed" || state.status === "failed") && e.type !== "log") return s;

  switch (e.type) {
    case "job.started":
      return {
        ...s,
        status: "running",
        files: e.files,
        combiner: e.combiner,
        modelVersion: e.modelVersion ?? null,
        prefilter: e.prefilter,
        totalPairs: e.totalPairs,
        candidatePairs: e.candidatePairs,
        total: e.candidatePairs,
      };
    case "pair.partial": {
      const prev = state.pairs[e.pairKey];
      if (prev !== undefined && prev.stage === "final") return s;
      const pair: PairState = {
        pairKey: e.pairKey,
        fileA: e.fileA,
        fileB: e.fileB,
        stage: "ast",
        astFeatures: e.astFeatures,
        astScore: e.astScore,
        neuralScore: null,
        score: null,
        threshold: null,
        verdict: null,
        detector: null,
        combinerVersion: null,
        matches: [],
        warnings: [],
        updatedSeq: e.seq,
      };
      return { ...s, pairs: { ...state.pairs, [e.pairKey]: pair } };
    }
    case "pair.scored": {
      const prev = state.pairs[e.pairKey];
      if (prev !== undefined && prev.stage === "final" && !newWins(prev, e, state.combiner?.version)) {
        return s;
      }
      const pair: PairState = {
        pairKey: e.pairKey,
        fileA: e.fileA,
        fileB: e.fileB,
        stage: "final",
        astFeatures: e.astFeatures,
        astScore: e.astScore,
        neuralScore: e.neuralScore ?? null,
        score: e.score,
        threshold: e.threshold,
        verdict: e.verdict,
        detector: e.detector,
        combinerVersion: e.combinerVersion,
        matches: e.matches ?? [],
        warnings: e.warnings ?? [],
        updatedSeq: e.seq,
      };
      const flagged = state.flagged - isFlagged(prev) + isFlagged(pair);
      return { ...s, flagged, pairs: { ...state.pairs, [e.pairKey]: pair } };
    }
    case "progress":
      return { ...s, done: e.done, total: e.total };
    case "log": {
      const line = { seq: e.seq, level: e.level ?? "info", message: e.message ?? "", ts: e.ts ?? "" };
      return { ...s, logs: [...state.logs, line].slice(-MAX_LOGS) };
    }
    case "job.completed":
      return { ...s, status: "completed", elapsed: e.seconds };
    case "job.failed":
      return { ...s, status: "failed", error: e.error };
    default:
      return s;
  }
}

/**
 * Apply a batch in seq order. The UI buffers events for ~250 ms and calls this once per
 * batch, so React re-renders a few times per second instead of once per pair.
 */
export function combineMany(state: ReportState, events: JobEvent[]): ReportState {
  return [...events].sort((a, b) => a.seq - b.seq).reduce(combineResults, state);
}
