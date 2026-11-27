"use client";
import { useCallback, useEffect, useState } from "react";
import { UploadDropzone } from "@/components/UploadDropzone";
import { ReportViewer } from "@/components/report/ReportViewer";

const JOB_PARAM = "job";
const JOB_ID_RE = /^[A-Za-z0-9_-]{1,128}$/;

/** Upload and report live on the same page; the job id is mirrored in ?job=<id>. */
export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);

  // Restore ?job= on load (read in an effect so the server render stays deterministic).
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get(JOB_PARAM);
    if (id && JOB_ID_RE.test(id)) setJobId(id);
  }, []);

  const selectJob = useCallback((id: string | null) => {
    setJobId(id);
    const url = new URL(window.location.href);
    if (id) url.searchParams.set(JOB_PARAM, id);
    else url.searchParams.delete(JOB_PARAM);
    window.history.replaceState(window.history.state, "", url);
  }, []);

  return (
    <main className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-2xl font-semibold">CodePlag</h1>
        <p className="text-sm text-muted-foreground">Similitud de código: AST + modelo neuronal</p>
      </header>
      {/* The upload collapses to a compact bar once a job exists; we never navigate away. */}
      <UploadDropzone compact={jobId !== null} onJobCreated={selectJob} onReset={() => selectJob(null)} />
      {jobId && <ReportViewer key={jobId} jobId={jobId} />}
    </main>
  );
}
