"use client";
import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

// The worker is copied to /public by scripts/copy-pdf-worker.mjs (same pdfjs-dist version as
// react-pdf), so it is served as a static file instead of being bundled by webpack.
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

export function PdfViewer({ url }: { url: string }) {
  const [pages, setPages] = useState(0);
  const [page, setPage] = useState(1);
  const [width, setWidth] = useState(800);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.min(1000, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div ref={box} className="space-y-2">
      {pages > 1 && (
        <div className="flex items-center gap-2 text-sm">
          <Button size="icon" variant="outline" aria-label="Página anterior" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="tabular-nums">
            Página {page} de {pages}
          </span>
          <Button
            size="icon"
            variant="outline"
            aria-label="Página siguiente"
            disabled={page >= pages}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
      <div className="overflow-auto rounded-md border border-border bg-muted/30">
        <Document
          file={url}
          onLoadSuccess={({ numPages }) => {
            setPages(numPages);
            setPage(1);
          }}
          loading={<p className="p-4 text-sm text-muted-foreground">Cargando PDF…</p>}
          error={<p className="p-4 text-sm text-destructive">No se pudo cargar el PDF del reporte.</p>}
        >
          <Page pageNumber={page} width={width} />
        </Document>
      </div>
    </div>
  );
}
