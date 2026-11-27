"use client";
import dynamic from "next/dynamic";
import { Download } from "lucide-react";
import { apiUrls } from "@/lib/api";
import type { JobStatus } from "@/lib/types";

// pdf.js needs browser APIs (DOMMatrix, canvas): load the viewer only on the client.
const PdfViewer = dynamic(() => import("./PdfViewer").then((m) => m.PdfViewer), {
  ssr: false,
  loading: () => <p className="text-sm text-muted-foreground">Cargando visor…</p>,
});

export function PdfTab({ jobId, status }: { jobId: string; status: JobStatus }) {
  if (status !== "completed") {
    return <p className="text-sm text-muted-foreground">El PDF estará disponible cuando termine el análisis.</p>;
  }
  const url = apiUrls.pdf(jobId);
  return (
    <div className="space-y-3">
      <a
        href={url}
        download={`codeplag-${jobId}.pdf`}
        className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-4 text-sm font-medium hover:bg-muted"
      >
        <Download className="h-4 w-4" /> Descargar PDF
      </a>
      <PdfViewer url={url} />
    </div>
  );
}
