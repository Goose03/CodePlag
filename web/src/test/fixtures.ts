// Test fixtures. The real goldens live in web/shared/ (generated from the Python code); the
// hand-written list below keeps the reducer tests meaningful even without them and covers
// cases a happy-path job does not produce (stale combiner version, late duplicate result).
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { CombinerParams, FileRef, JobEvent } from "@/lib/types";

// (path + __dirname instead of import.meta.url: jsdom replaces the global URL class.)
export const SHARED_DIR = resolve(__dirname, "../../shared");
export const sharedPath = (name: string) => join(SHARED_DIR, name);

/** Parsed JSON from web/shared/<name>, or null when the file does not exist yet. */
export function readShared<T>(name: string): T | null {
  const p = sharedPath(name);
  return existsSync(p) ? (JSON.parse(readFileSync(p, "utf-8")) as T) : null;
}

export const SAMPLE_COMBINER: CombinerParams = {
  name: "combined",
  method: "logistic",
  features: ["ted", "subtree", "ngram", "neural"],
  coef: [0.228, 3.596, 3.357, 5.153],
  intercept: -4.519,
  threshold: 0.1766,
  version: "v-current",
};

const sha = (c: string) => c.repeat(64);
export const SAMPLE_FILES: FileRef[] = [
  { fileId: sha("a"), name: "a.py", language: "python", sha256: sha("a") },
  { fileId: sha("b"), name: "b.py", language: "python", sha256: sha("b") },
  { fileId: sha("c"), name: "c.py", language: "python", sha256: sha("c") },
];

const JOB = "sample-job";
const ts = "2026-01-01T00:00:00+00:00";

export const SAMPLE_EVENTS: JobEvent[] = [
  {
    type: "job.started",
    seq: 1,
    jobId: JOB,
    files: SAMPLE_FILES,
    totalPairs: 3,
    candidatePairs: 3,
    prefilter: "none",
    combiner: SAMPLE_COMBINER,
    modelVersion: "best.pt",
  },
  { type: "log", seq: 2, jobId: JOB, level: "info", message: "3 archivos, 3 pares candidatos.", ts },
  {
    type: "pair.partial",
    seq: 3,
    jobId: JOB,
    stage: "ast",
    pairKey: "ab",
    fileA: "a.py",
    fileB: "b.py",
    astFeatures: { ted: 1, subtree: 1, ngram: 1 },
    astScore: 1,
  },
  {
    type: "pair.partial",
    seq: 4,
    jobId: JOB,
    stage: "ast",
    pairKey: "ac",
    fileA: "a.py",
    fileB: "c.py",
    astFeatures: { ted: 0.2, subtree: 0.1, ngram: 0.1 },
    astScore: 0.13,
  },
  {
    type: "pair.scored",
    seq: 5,
    jobId: JOB,
    stage: "final",
    pairKey: "ab",
    combinerVersion: "v-current",
    fileA: "a.py",
    fileB: "b.py",
    astFeatures: { ted: 1, subtree: 1, ngram: 1 },
    astScore: 1,
    neuralScore: 0.98,
    score: 0.9995,
    threshold: 0.1766,
    verdict: "likely_plagiarism",
    detector: "combined",
    matches: [{ linesA: [1, 20], linesB: [2, 21], size: 80, node: "module" }],
    warnings: [],
  },
  {
    type: "pair.scored",
    seq: 6,
    jobId: JOB,
    stage: "final",
    pairKey: "ac",
    combinerVersion: "v-current",
    fileA: "a.py",
    fileB: "c.py",
    astFeatures: { ted: 0.2, subtree: 0.1, ngram: 0.1 },
    astScore: 0.13,
    neuralScore: 0.1,
    score: 0.03,
    threshold: 0.1766,
    verdict: "unlikely_plagiarism",
    detector: "combined",
    matches: [],
    warnings: [],
  },
  { type: "progress", seq: 7, jobId: JOB, done: 2, total: 3 },
  // A late result for "ac" computed with an older combiner: must NOT replace the current one.
  {
    type: "pair.scored",
    seq: 8,
    jobId: JOB,
    stage: "final",
    pairKey: "ac",
    combinerVersion: "v-old",
    fileA: "a.py",
    fileB: "c.py",
    astFeatures: { ted: 0.2, subtree: 0.1, ngram: 0.1 },
    astScore: 0.13,
    neuralScore: 0.1,
    score: 0.6,
    threshold: 0.5,
    verdict: "likely_plagiarism",
    detector: "combined",
    matches: [],
    warnings: [],
  },
  {
    type: "pair.partial",
    seq: 9,
    jobId: JOB,
    stage: "ast",
    pairKey: "bc",
    fileA: "b.py",
    fileB: "c.py",
    astFeatures: { ted: 0.3, subtree: 0.2, ngram: 0.1 },
    astScore: 0.2,
  },
  {
    type: "pair.scored",
    seq: 10,
    jobId: JOB,
    stage: "final",
    pairKey: "bc",
    combinerVersion: "v-current",
    fileA: "b.py",
    fileB: "c.py",
    astFeatures: { ted: 0.3, subtree: 0.2, ngram: 0.1 },
    astScore: 0.2,
    neuralScore: null,
    score: 0.04,
    threshold: 0.1766,
    verdict: "unlikely_plagiarism",
    detector: "combined",
    matches: [],
    warnings: ["Neural model unavailable for this pair; AST-only score."],
  },
  { type: "progress", seq: 11, jobId: JOB, done: 3, total: 3 },
  { type: "log", seq: 12, jobId: JOB, level: "warning", message: "1 par sin score neuronal.", ts },
  { type: "job.completed", seq: 13, jobId: JOB, seconds: 1.5, flagged: 1 },
];

export const GOLDEN_EVENTS = readShared<JobEvent[]>("events_golden.json");

/** Datasets the reducer properties are checked against. */
export const EVENT_DATASETS: [string, JobEvent[]][] = [
  ["muestra escrita a mano", SAMPLE_EVENTS],
  ...(GOLDEN_EVENTS ? ([["events_golden.json (job real)", GOLDEN_EVENTS]] as [string, JobEvent[]][]) : []),
];
