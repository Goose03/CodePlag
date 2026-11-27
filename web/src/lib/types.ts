// Wire contract with the API. Mirror of web/api/schemas.py and web/api/reducer.py.
// Changing a field here means changing it there too (the parity tests will fail otherwise).

export type Verdict = "likely_plagiarism" | "unlikely_plagiarism";

/** Parameters of the fitted score combiner (codeplag.similarity.combiner.Combiner). */
export interface CombinerParams {
  name: string; // "combined" | "ast_only" | "neural_only" | "weighted" | "ast_weighted"
  method: "logistic" | "weighted";
  features: string[]; // order of `coef`
  coef: number[];
  intercept: number;
  threshold: number; // calibrated on validation for FPR <= 5 %
  version: string; // short sha256 of the parameters
}

export interface FileRef {
  fileId: string; // sha256 of the content; GET /api/jobs/:id/files/:fileId
  name: string;
  language: string;
  sha256: string;
}

export interface MatchRegion {
  linesA: [number, number];
  linesB: [number, number];
  size: number;
  node: string;
}

interface Base {
  seq: number; // strictly increasing per job, starts at 1
  jobId?: string;
}

export type JobStarted = Base & {
  type: "job.started";
  files: FileRef[];
  totalPairs: number;
  candidatePairs: number;
  prefilter: string;
  combiner: CombinerParams;
  modelVersion: string | null;
};

export type PairPartial = Base & {
  type: "pair.partial";
  stage: "ast";
  pairKey: string;
  fileA: string;
  fileB: string;
  astFeatures: Record<string, number>;
  astScore: number;
};

export type PairScored = Base & {
  type: "pair.scored";
  stage: "final";
  pairKey: string;
  combinerVersion: string;
  fileA: string;
  fileB: string;
  languageA?: string;
  languageB?: string;
  astFeatures: Record<string, number>;
  astScore: number;
  neuralScore: number | null;
  score: number;
  threshold: number;
  verdict: Verdict;
  detector: string;
  matches: MatchRegion[];
  warnings: string[];
};

export type Progress = Base & { type: "progress"; done: number; total: number };
export type LogEvent = Base & {
  type: "log";
  level: "debug" | "info" | "warning" | "error";
  message: string;
  ts: string;
};
export type JobCompleted = Base & { type: "job.completed"; seconds: number; flagged: number };
export type JobFailed = Base & { type: "job.failed"; error: string; retryable: boolean };

export type JobEvent =
  | JobStarted
  | PairPartial
  | PairScored
  | Progress
  | LogEvent
  | JobCompleted
  | JobFailed;

export const EVENT_TYPES: JobEvent["type"][] = [
  "job.started",
  "pair.partial",
  "pair.scored",
  "progress",
  "log",
  "job.completed",
  "job.failed",
];

export interface PairState {
  pairKey: string;
  fileA: string;
  fileB: string;
  stage: "ast" | "final";
  astFeatures: Record<string, number>;
  astScore: number;
  neuralScore: number | null;
  score: number | null;
  threshold: number | null;
  verdict: Verdict | null;
  detector: string | null;
  combinerVersion: string | null;
  matches: MatchRegion[];
  warnings: string[];
  updatedSeq: number;
}

export interface LogLine {
  seq: number;
  level: string;
  message: string;
  ts: string;
}

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface ReportState {
  jobId: string;
  status: JobStatus;
  lastSeq: number;
  gap: boolean;
  files: FileRef[];
  combiner: CombinerParams | null;
  modelVersion: string | null;
  prefilter: string | null;
  totalPairs: number;
  candidatePairs: number;
  done: number;
  total: number;
  pairs: Record<string, PairState>;
  flagged: number;
  elapsed: number | null;
  error: string | null;
  logs: LogLine[];
}

export interface CreateJobResponse {
  jobId: string;
  status: "queued" | "exists";
  deduplicated: boolean;
}
