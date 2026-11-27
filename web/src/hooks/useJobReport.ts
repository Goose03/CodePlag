"use client";
// Live report state: snapshot first (GET /report), then SSE from its lastSeq (§7.3).
// The reducer drops repeated events, so overlap between snapshot and stream is harmless.
import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiUrls, errorMessage } from "@/lib/api";
import { combineMany, initialState } from "@/lib/report-reducer";
import { EVENT_TYPES, type JobEvent, type JobStatus, type ReportState } from "@/lib/types";

export const FLUSH_MS = 250;
export const POLL_MS = 2000;
export const MAX_SSE_FAILURES = 3;

export const reportKey = (jobId: string) => ["report", jobId] as const;
export const isTerminal = (s: JobStatus | undefined) => s === "completed" || s === "failed";

/** GET /report. 404 = the job has not emitted any event yet (or expired) -> empty state. */
export async function fetchReport(jobId: string, signal?: AbortSignal): Promise<ReportState> {
  const r = await fetch(apiUrls.report(jobId), { signal });
  if (r.status === 404) return initialState(jobId);
  if (!r.ok) throw new Error(errorMessage(await r.text(), `HTTP ${r.status}`));
  return (await r.json()) as ReportState;
}

export type Transport = "sse" | "polling" | "idle";

export function useJobReport(jobId: string) {
  const qc = useQueryClient();
  const key = useMemo(() => reportKey(jobId), [jobId]);
  // Polling is only the fallback when SSE keeps failing (corporate proxies, etc.).
  const [polling, setPolling] = useState(false);

  const query = useQuery<ReportState>({
    queryKey: key,
    queryFn: async ({ signal }) => {
      const snap = await fetchReport(jobId, signal);
      // The server snapshot wins, except when it is older than what the stream already gave
      // us and we have no gap to repair (avoids flicker from a slow in-flight request).
      const cur = qc.getQueryData<ReportState>(key);
      if (cur && !cur.gap && cur.lastSeq > snap.lastSeq) return cur;
      return snap;
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchInterval: (q) => (polling && !isTerminal(q.state.data?.status) ? POLL_MS : false),
  });

  const ready = query.isSuccess;
  useEffect(() => {
    if (!ready || polling) return;
    const start = qc.getQueryData<ReportState>(key);
    if (isTerminal(start?.status)) return;
    if (typeof EventSource === "undefined") {
      setPolling(true);
      return;
    }

    // EventSource cannot set Last-Event-ID on the first connection, so we pass ?after=; on
    // automatic reconnections the browser sends the header and the server prefers it.
    const es = new EventSource(apiUrls.events(jobId, start?.lastSeq ?? 0));
    let failures = 0;
    let buffer: JobEvent[] = [];

    const flush = () => {
      if (!buffer.length) return;
      const batch = buffer;
      buffer = [];
      const prev = qc.getQueryData<ReportState>(key) ?? initialState(jobId);
      const next = combineMany(prev, batch);
      if (next === prev) return;
      qc.setQueryData<ReportState>(key, next);
      // Hole in seq -> the server snapshot replaces our state (only on the transition).
      if (next.gap && !prev.gap) void qc.invalidateQueries({ queryKey: key });
    };
    const timer = setInterval(flush, FLUSH_MS);

    const onEvent = (m: MessageEvent<string>) => {
      failures = 0;
      let ev: JobEvent;
      try {
        ev = JSON.parse(m.data) as JobEvent;
      } catch {
        return;
      }
      buffer.push(ev);
      if (ev.type === "job.completed" || ev.type === "job.failed") {
        flush();
        es.close();
      }
    };
    EVENT_TYPES.forEach((t) => es.addEventListener(t, onEvent as EventListener));

    es.onerror = () => {
      failures += 1;
      // CLOSED means the browser gave up (e.g. HTTP error status): no point in waiting.
      if (failures >= MAX_SSE_FAILURES || es.readyState === EventSource.CLOSED) {
        es.close();
        flush();
        setPolling(true);
      }
    };

    return () => {
      clearInterval(timer);
      es.close();
      flush();
    };
  }, [ready, polling, jobId, key, qc]);

  const status = query.data?.status;
  const transport: Transport = isTerminal(status) ? "idle" : polling ? "polling" : "sse";
  return { ...query, transport };
}
