"use client";
import { useCallback, useState } from "react";
import { useDropzone, type FileError, type FileRejection } from "react-dropzone";
import { FileCode2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useUpload } from "@/hooks/useUpload";
import { cn } from "@/lib/utils";

// Must match codeplag.ast_analysis.parser.EXTENSIONS and WebSettings. The server validates
// again: the client-side checks only give fast feedback.
export const EXTENSIONS = [".py", ".pyw", ".java", ".js", ".mjs", ".cjs"] as const;
export const MAX_FILE_BYTES = 512 * 1024;
export const MAX_TOTAL_BYTES = 20 * 1024 * 1024;
export const MIN_FILES = 2;
export const MAX_FILES = 200;

const extOf = (name: string) => {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i).toLowerCase();
};
const fileKey = (f: File) => `${f.name}:${f.size}:${f.lastModified}`;
const fmtKB = (bytes: number) => `${(bytes / 1024).toFixed(1)} KB`;

/** Returns Spanish error(s) for a file, or null when valid. */
export function validateFile(f: File): FileError | null {
  if (!(EXTENSIONS as readonly string[]).includes(extOf(f.name))) {
    return { code: "file-invalid-type", message: `extensión no soportada (usa ${EXTENSIONS.join(" ")})` };
  }
  if (f.size > MAX_FILE_BYTES) {
    return { code: "file-too-large", message: `pesa ${fmtKB(f.size)}, el máximo es 512 KB` };
  }
  if (f.size === 0) return { code: "file-empty", message: "está vacío" };
  return null;
}

export function UploadDropzone({
  onJobCreated,
  onReset,
  compact,
}: {
  onJobCreated: (jobId: string) => void;
  onReset?: () => void;
  compact: boolean;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [info, setInfo] = useState<string | null>(null);
  const { upload, cancel, progress, error: uploadError } = useUpload();
  const uploading = progress !== null;

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      const errs = rejected.map((r) => `${r.file.name}: ${r.errors.map((e) => e.message).join(", ")}`);
      // Dedupe by name+size+mtime so dropping the same file twice does not duplicate it.
      const seen = new Set(files.map(fileKey));
      const fresh = accepted.filter((f) => !seen.has(fileKey(f)) && seen.add(fileKey(f)));
      const next = [...files, ...fresh];
      if (next.length > MAX_FILES) {
        errs.push(`Máximo ${MAX_FILES} archivos: se ignoraron ${next.length - MAX_FILES}.`);
      }
      setFiles(next.slice(0, MAX_FILES));
      setErrors(errs);
      setInfo(null);
    },
    [files],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    validator: validateFile,
    multiple: true,
    disabled: uploading,
  });

  const totalBytes = files.reduce((a, f) => a + f.size, 0);
  const tooBig = totalBytes > MAX_TOTAL_BYTES;
  const canSubmit = files.length >= MIN_FILES && files.length <= MAX_FILES && !tooBig && !uploading;

  async function submit() {
    setErrors([]);
    setInfo(null);
    const res = await upload(files);
    if (!res) return;
    setFiles([]);
    if (res.deduplicated) setInfo("Ya existía un análisis con estos mismos archivos: se muestra su reporte.");
    onJobCreated(res.jobId);
  }

  const shownErrors = [...errors, ...(uploadError ? [uploadError] : [])];
  if (tooBig) shownErrors.push(`El total (${fmtKB(totalBytes)}) supera el límite de 20 MB.`);

  return (
    <section aria-label="Subir archivos" className="space-y-3 rounded-xl border border-border p-4">
      <div
        {...getRootProps()}
        className={cn(
          "cursor-pointer rounded-lg border-2 border-dashed border-border text-center transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
          compact && files.length === 0 ? "p-3" : "p-8",
          isDragActive && "border-primary bg-primary/5",
          uploading && "cursor-not-allowed opacity-60",
        )}
      >
        <input {...getInputProps({ accept: EXTENSIONS.join(",") })} aria-label="Archivos de código" />
        <div className="flex flex-col items-center gap-1">
          <Upload className="h-5 w-5 text-muted-foreground" aria-hidden />
          <p className="font-medium">
            {isDragActive
              ? "Suelta los archivos aquí"
              : compact
                ? "Arrastra archivos para un nuevo análisis (o haz clic)"
                : "Arrastra archivos .py, .java o .js (o haz clic para elegirlos)"}
          </p>
          <p className="text-sm text-muted-foreground">
            Entre {MIN_FILES} y {MAX_FILES} archivos, 512 KB c/u ({EXTENSIONS.join(" ")}). El código nunca se ejecuta.
          </p>
        </div>
      </div>

      {files.length > 0 && (
        <ul aria-label="Archivos seleccionados" className="max-h-48 divide-y divide-border overflow-auto rounded-md border border-border text-sm">
          {files.map((f) => (
            <li key={fileKey(f)} className="flex items-center justify-between gap-2 px-3 py-1.5">
              <span className="flex min-w-0 items-center gap-2">
                <FileCode2 className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                <span className="truncate">{f.name}</span>
                <span className="shrink-0 text-xs text-muted-foreground">{fmtKB(f.size)}</span>
              </span>
              <Button
                variant="ghost"
                size="icon"
                disabled={uploading}
                onClick={() => setFiles((prev) => prev.filter((x) => x !== f))}
                aria-label={`Quitar ${f.name}`}
              >
                <X className="h-4 w-4" />
              </Button>
            </li>
          ))}
        </ul>
      )}

      {shownErrors.map((e, i) => (
        <p key={`${i}:${e}`} role="alert" className="text-sm text-destructive">
          {e}
        </p>
      ))}
      {info && <p className="text-sm text-muted-foreground">{info}</p>}

      {uploading && (
        <div className="space-y-1">
          <Progress value={progress} aria-label="Progreso de subida" />
          <p className="text-xs text-muted-foreground">Subiendo… {progress}%</p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={!canSubmit} onClick={submit}>
          Analizar {files.length} {files.length === 1 ? "archivo" : "archivos"}
        </Button>
        {uploading && (
          <Button variant="outline" onClick={cancel}>
            Cancelar
          </Button>
        )}
        {files.length > 0 && !uploading && (
          <Button variant="ghost" onClick={() => setFiles([])}>
            Vaciar lista
          </Button>
        )}
        {compact && onReset && !uploading && (
          <Button variant="ghost" onClick={onReset}>
            Cerrar reporte
          </Button>
        )}
        {files.length === 1 && <span className="text-sm text-muted-foreground">Agrega al menos un archivo más.</span>}
      </div>
    </section>
  );
}
