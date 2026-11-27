"""Upload validation. Uploaded code is never executed: tree-sitter only parses it."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from fastapi import HTTPException, UploadFile

from codeplag.ast_analysis.parser import EXTENSIONS

from .settings import WebSettings

JOB_ID_RE = re.compile(r"^[0-9a-f]{20}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidFile:
    """A validated upload.

    Attributes:
        name: Base name only (directories are stripped to prevent path traversal).
        language: Language detected from the extension.
        code: Decoded UTF-8 source.
        sha256: Hash of the raw bytes; also the on-disk file name and the ``fileId``.
    """

    name: str
    language: str
    code: str
    sha256: str


def safe_name(filename: str | None) -> str:
    """Base name of an uploaded file, whatever path separator the client used."""
    return PureWindowsPath(PurePosixPath(filename or "").name).name


def check_job_id(job_id: str) -> str:
    """Reject anything that is not a job id (they are also directory names)."""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job id.")
    return job_id


async def validate_uploads(files: list[UploadFile], cfg: WebSettings) -> list[ValidFile]:
    """Validate count, extension, size and encoding of the uploads.

    Raises:
        HTTPException: 413/415/422 with a message the UI shows as-is.
    """
    if not 2 <= len(files) <= cfg.max_files:
        raise HTTPException(422, f"Upload between 2 and {cfg.max_files} files.")
    out: list[ValidFile] = []
    seen: set[str] = set()
    total = 0
    for upload in files:
        name = safe_name(upload.filename)
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in EXTENSIONS:
            allowed = ", ".join(sorted(EXTENSIONS))
            raise HTTPException(415, f"{name}: unsupported extension (allowed: {allowed}).")
        # Read at most one byte past the limit so huge files are rejected without loading them.
        raw = await upload.read(cfg.max_file_bytes + 1)
        if len(raw) > cfg.max_file_bytes:
            raise HTTPException(413, f"{name}: larger than {cfg.max_file_bytes // 1024} KB.")
        total += len(raw)
        if total > cfg.max_total_bytes:
            raise HTTPException(413, "Total upload size exceeds the limit.")
        try:
            code = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(422, f"{name}: not UTF-8 text.") from None
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:  # identical content uploaded twice: comparing it is pointless
            continue
        seen.add(digest)
        out.append(ValidFile(name, EXTENSIONS[suffix], code, digest))
    if len(out) < 2:
        raise HTTPException(422, "Need at least 2 distinct files.")
    return out
