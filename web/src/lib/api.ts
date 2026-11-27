// HTTP helpers for the CodePlag API (contract: docs/architecture/web-report-viewer.md §3).
export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

const job = (jobId: string) => `${API_URL}/api/jobs/${encodeURIComponent(jobId)}`;

export const apiUrls = {
  createJob: () => `${API_URL}/api/jobs`,
  report: (jobId: string) => `${job(jobId)}/report`,
  events: (jobId: string, after: number) => `${job(jobId)}/events?after=${after}`,
  file: (jobId: string, fileId: string) => `${job(jobId)}/files/${encodeURIComponent(fileId)}`,
  pdf: (jobId: string) => `${job(jobId)}/report.pdf`,
};

/** Human-readable message from a FastAPI error body (`{detail: string | ValidationError[]}`). */
export function errorMessage(body: string, fallback: string): string {
  try {
    const parsed: unknown = JSON.parse(body);
    const detail = (parsed as { detail?: unknown } | null)?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (d as { msg?: unknown } | null)?.msg)
        .filter((m): m is string => typeof m === "string");
      if (msgs.length) return msgs.join("; ");
    }
  } catch {
    /* not JSON */
  }
  return body.trim() || fallback;
}

/** GET a text resource; throws with the API's `detail` on non-2xx. */
export async function fetchText(url: string, signal?: AbortSignal): Promise<string> {
  const r = await fetch(url, { signal });
  const text = await r.text();
  if (!r.ok) throw new Error(errorMessage(text, `HTTP ${r.status}`));
  return text;
}
