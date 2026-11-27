// Copies the pdf.js worker that matches react-pdf's pdfjs-dist into public/, so the browser
// loads it as a static file (/pdf.worker.min.mjs). Bundling the .mjs worker through webpack
// in Next 14 breaks on minification, so we serve it as-is. Runs before `dev` and `build`.
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(root, "package.json"));
const fromReactPdf = createRequire(require.resolve("react-pdf"));
const pdfjsDir = dirname(fromReactPdf.resolve("pdfjs-dist/package.json"));
const src = join(pdfjsDir, "build", "pdf.worker.min.mjs");
mkdirSync(join(root, "public"), { recursive: true });
copyFileSync(src, join(root, "public", "pdf.worker.min.mjs"));
console.log(`pdf.js worker copied from ${src}`);
