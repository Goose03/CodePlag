"use client";
import { useCallback, useRef, useState } from "react";
import { apiUrls, errorMessage } from "@/lib/api";
import type { CreateJobResponse } from "@/lib/types";

const abortError = () => new DOMException("Subida cancelada", "AbortError");

/** POST /api/jobs as multipart (`files`). fetch() has no upload progress; XMLHttpRequest does. */
export function uploadFiles(
  files: File[],
  onProgress: (pct: number) => void,
  signal?: AbortSignal,
): Promise<CreateJobResponse> {
  return new Promise<CreateJobResponse>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status === 202) {
        try {
          resolve(JSON.parse(xhr.responseText) as CreateJobResponse);
        } catch {
          reject(new Error("Respuesta inválida del servidor."));
        }
        return;
      }
      reject(new Error(errorMessage(xhr.responseText, `Error ${xhr.status} ${xhr.statusText}`.trim())));
    };
    xhr.onerror = () => reject(new Error("Error de red: no se pudo contactar la API."));
    xhr.onabort = () => reject(abortError());
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.open("POST", apiUrls.createJob());
    xhr.send(form);
  });
}

/** Stateful wrapper: progress (0-100, or null when idle), last error, and cancel(). */
export function useUpload() {
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  const upload = useCallback(async (files: File[]): Promise<CreateJobResponse | null> => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setError(null);
    setProgress(0);
    try {
      return await uploadFiles(files, setProgress, c.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
      return null;
    } finally {
      if (ctrl.current === c) ctrl.current = null;
      setProgress(null);
    }
  }, []);

  const cancel = useCallback(() => ctrl.current?.abort(), []);
  return { upload, cancel, progress, error };
}
