"""CodePlag web API: uploads, background jobs and live (SSE) reports.

See ``docs/architecture/web-report-viewer.md`` for the design. The API process never loads
PyTorch; only the worker (:mod:`api.worker`) runs the detector.
"""
