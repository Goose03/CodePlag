import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes, letting later ones win (shadcn convention). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export const fmtScore = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits);
