"""PDF version of a finished report (ReportLab: pure Python, no system libraries).

The PDF always uses the calibrated threshold (never the browser's what-if value) and always
includes the disclaimer.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DISCLAIMER_ES = (
    "Un score alto es evidencia para revisión humana, no prueba de plagio. Plantillas, código "
    "base y algoritmos estándar pueden parecerse legítimamente."
)
MAX_ROWS = 500


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_pdf(state: dict[str, Any]) -> bytes:
    """Render the report state (see :mod:`api.reducer`) as a PDF document."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"CodePlag {state['jobId']}",
    )
    styles = getSampleStyleSheet()
    combiner = state.get("combiner") or {}
    pairs = sorted(
        state["pairs"].values(),
        key=lambda p: -(p["score"] if p["score"] is not None else p["astScore"]),
    )
    story: list[Any] = [
        Paragraph("Reporte CodePlag", styles["Title"]),
        Paragraph(
            escape(
                f"Job {state['jobId']} · {len(state['files'])} archivos · "
                f"{state['candidatePairs']} pares candidatos de {state['totalPairs']} "
                f"(prefiltro: {state.get('prefilter') or '-'}) · "
                f"{state['flagged']} pares marcados · "
                f"combiner '{combiner.get('name', '-')}' con umbral "
                f"{_fmt(combiner.get('threshold'))} · "
                f"{_fmt(state.get('elapsed'))} s"
            ),
            styles["Normal"],
        ),
        Spacer(1, 0.4 * cm),
    ]
    header = ["#", "Archivo A", "Archivo B", "Score", "Veredicto", "AST", "Neural", "Detector"]
    rows: list[list[str]] = [header]
    for i, p in enumerate(pairs[:MAX_ROWS], start=1):
        verdict = {
            "likely_plagiarism": "PROBABLE PLAGIO",
            "unlikely_plagiarism": "improbable",
        }.get(p["verdict"] or "", "pendiente")
        rows.append(
            [
                str(i),
                p["fileA"],
                p["fileB"],
                _fmt(p["score"]),
                verdict,
                _fmt(p["astScore"]),
                _fmt(p["neuralScore"]),
                p["detector"] or "-",
            ]
        )
    table = Table(
        rows,
        repeatRows=1,
        colWidths=[1 * cm, 6.5 * cm, 6.5 * cm, 2 * cm, 3.5 * cm, 2 * cm, 2 * cm, 3 * cm],
    )
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for r, p in enumerate(pairs[:MAX_ROWS], start=1):
        if p["verdict"] == "likely_plagiarism":
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fee2e2")))
    table.setStyle(TableStyle(style))
    story.append(table)
    if len(pairs) > MAX_ROWS:
        story.append(
            Paragraph(f"(Se muestran {MAX_ROWS} de {len(pairs)} pares.)", styles["Italic"])
        )
    story += [Spacer(1, 0.5 * cm), Paragraph(escape(DISCLAIMER_ES), styles["Italic"])]
    doc.build(story)
    return buf.getvalue()
