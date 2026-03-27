"""
report.py -- PDF Report Generator for ECG AI Analyzer

Generates a professional single-page PDF report containing:
    - ECG signal snapshot (embedded as image)
    - Heart rate & HRV metrics
    - Detection performance (Precision / Recall / F1)
    - Clinical findings with severity badges
    - Disclaimer footer

Uses ReportLab for layout.  All data is passed in as plain Python
types so this module has zero coupling to Streamlit or Plotly.
"""

import io
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


# ── Color palette ─────────────────────────────────────────────────────

_BG_DARK    = colors.HexColor("#0F1923")
_PANEL      = colors.HexColor("#182633")
_ACCENT     = colors.HexColor("#00D4AA")
_RED        = colors.HexColor("#EF4444")
_YELLOW     = colors.HexColor("#FBBF24")
_ORANGE     = colors.HexColor("#F97316")
_GREEN      = colors.HexColor("#22C55E")
_TEXT_MAIN  = colors.HexColor("#E2E8F0")
_TEXT_DIM   = colors.HexColor("#94A3B8")
_DIVIDER    = colors.HexColor("#2D4A5E")
_WHITE      = colors.white
_BLACK      = colors.black


# ── Styles ────────────────────────────────────────────────────────────

def _build_styles():
    """Build paragraph styles for the report."""
    ss = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RPT_Title", parent=ss["Title"],
        fontSize=22, leading=26,
        textColor=_ACCENT, fontName="Helvetica-Bold",
        spaceAfter=2 * mm,
    )
    subtitle_style = ParagraphStyle(
        "RPT_Subtitle", parent=ss["Normal"],
        fontSize=10, leading=13,
        textColor=colors.HexColor("#64748B"),
        fontName="Helvetica",
        spaceAfter=6 * mm,
    )
    section_style = ParagraphStyle(
        "RPT_Section", parent=ss["Heading2"],
        fontSize=13, leading=16,
        textColor=colors.HexColor("#0F1923"),
        fontName="Helvetica-Bold",
        spaceBefore=5 * mm, spaceAfter=3 * mm,
        borderPadding=(0, 0, 1, 0),
    )
    body_style = ParagraphStyle(
        "RPT_Body", parent=ss["Normal"],
        fontSize=9.5, leading=13,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica",
    )
    metric_label = ParagraphStyle(
        "RPT_MetricLabel", parent=ss["Normal"],
        fontSize=9, leading=12,
        textColor=colors.HexColor("#64748B"),
        fontName="Helvetica",
    )
    metric_value = ParagraphStyle(
        "RPT_MetricValue", parent=ss["Normal"],
        fontSize=10, leading=13,
        textColor=colors.HexColor("#0F1923"),
        fontName="Helvetica-Bold",
    )
    disclaimer_style = ParagraphStyle(
        "RPT_Disclaimer", parent=ss["Normal"],
        fontSize=7, leading=9,
        textColor=colors.HexColor("#94A3B8"),
        fontName="Helvetica-Oblique",
        alignment=TA_CENTER,
        spaceBefore=4 * mm,
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "section": section_style,
        "body": body_style,
        "label": metric_label,
        "value": metric_value,
        "disclaimer": disclaimer_style,
    }


# ── Table helpers ─────────────────────────────────────────────────────

def _metrics_table(rows: List[Tuple[str, str]], col_widths=None):
    """Build a styled two-column key/value table."""
    if col_widths is None:
        col_widths = [55 * mm, 40 * mm]

    styles = _build_styles()

    data = []
    for label, value in rows:
        data.append([
            Paragraph(label, styles["label"]),
            Paragraph(value, styles["value"]),
        ])

    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#E2E8F0")),
    ]))
    return t


def _severity_color(severity: str) -> str:
    """Return hex color for a severity level."""
    return {
        "info": "#94A3B8",
        "mild": "#FBBF24",
        "moderate": "#F97316",
        "significant": "#EF4444",
    }.get(severity, "#94A3B8")


# ── Public API ────────────────────────────────────────────────────────

def generate_pdf_report(
    record_id: str,
    fs: float,
    n_samples: int,
    # Metrics
    mean_hr: float,
    median_hr: float,
    sdnn_ms: float,
    rmssd_ms: float,
    pnn50: float,
    peak_width_mean_ms: float,
    peak_width_std_ms: float,
    n_detected: int,
    # Eval (optional)
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1: Optional[float] = None,
    n_reference: Optional[int] = None,
    # Findings
    is_normal: bool = True,
    findings: Optional[List[Dict]] = None,
    summary: str = "",
    # ECG image (PNG bytes or path)
    ecg_image_bytes: Optional[bytes] = None,
) -> bytes:
    """Generate a professional PDF report and return raw bytes.

    Parameters
    ----------
    record_id : str
        Record identifier.
    fs : float
        Sampling frequency.
    n_samples : int
        Total signal length.
    mean_hr, median_hr : float
        Heart rate statistics.
    sdnn_ms, rmssd_ms, pnn50 : float
        HRV metrics (SDNN and RMSSD in ms).
    peak_width_mean_ms, peak_width_std_ms : float
        Peak width at half-prominence stats.
    n_detected : int
        Number of detected R-peaks.
    precision, recall, f1 : float or None
        Detection performance metrics.
    n_reference : int or None
        Number of expert-annotated beats.
    is_normal : bool
        Whether the rhythm is classified as normal.
    findings : list of dict or None
        Each dict has keys: name, severity, line1, line2.
    summary : str
        One-paragraph summary.
    ecg_image_bytes : bytes or None
        PNG image of the ECG signal.

    Returns
    -------
    bytes
        PDF file content.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )

    styles = _build_styles()
    story = []

    # ── Header ──────────────────────────────────────────────────────
    story.append(Paragraph("ECG AI Analyzer Report", styles["title"]))

    duration_sec = n_samples / fs if fs > 0 else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(
        f"Record: {record_id}  |  Fs: {fs:.0f} Hz  |  "
        f"Duration: {duration_sec:.1f} s  |  Generated: {now}",
        styles["subtitle"],
    ))

    story.append(HRFlowable(
        width="100%", thickness=0.8,
        color=colors.HexColor("#CBD5E1"),
        spaceAfter=4 * mm,
    ))

    # ── ECG Snapshot ────────────────────────────────────────────────
    if ecg_image_bytes is not None:
        story.append(Paragraph("ECG Signal Snapshot", styles["section"]))
        img_buf = io.BytesIO(ecg_image_bytes)
        img = Image(img_buf, width=170 * mm, height=55 * mm)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 4 * mm))

    # ── Metrics: side by side ───────────────────────────────────────
    story.append(Paragraph("Heart Rate &amp; HRV Metrics", styles["section"]))

    hrv_rows = [
        ("Mean HR", f"{mean_hr:.1f} BPM"),
        ("Median HR", f"{median_hr:.1f} BPM"),
        ("SDNN", f"{sdnn_ms:.1f} ms"),
        ("RMSSD", f"{rmssd_ms:.1f} ms"),
        ("pNN50", f"{pnn50:.1f} %"),
        ("Peak Width*", f"{peak_width_mean_ms:.1f} +/- {peak_width_std_ms:.1f} ms"),
    ]

    perf_rows = [
        ("Detected Beats", f"{n_detected:,}"),
    ]
    if n_reference is not None:
        perf_rows.append(("Expert Beats", f"{n_reference:,}"))
    if precision is not None:
        perf_rows.append(("Precision", f"{precision:.4f}"))
    if recall is not None:
        perf_rows.append(("Recall", f"{recall:.4f}"))
    if f1 is not None:
        perf_rows.append(("F1 Score", f"{f1:.4f}"))

    t_hrv = _metrics_table(hrv_rows)
    t_perf = _metrics_table(perf_rows)

    # Two-column layout
    outer = Table(
        [[t_hrv, t_perf]],
        colWidths=[95 * mm, 80 * mm],
        hAlign="LEFT",
    )
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(outer)

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<i>* Peak Width = approx. width at half-prominence "
        "(signal-processing proxy, not clinical QRS delineation).</i>",
        ParagraphStyle("fn", parent=styles["body"],
                       fontSize=7, textColor=colors.HexColor("#94A3B8")),
    ))

    # ── Clinical Findings ───────────────────────────────────────────
    story.append(Paragraph("Clinical Findings", styles["section"]))

    if is_normal:
        story.append(Paragraph(
            '<font color="#22C55E"><b>NORMAL</b></font> '
            "-- No rule-based abnormalities detected. "
            "Rhythm appears regular within normal limits.",
            styles["body"],
        ))
    elif findings:
        for f in findings:
            sev = f.get("severity", "INFO")
            sc = _severity_color(sev.lower())
            name = f.get("title", f.get("name", ""))
            l1 = f.get("line1", "")
            l2 = f.get("line2", "")
            story.append(Paragraph(
                f'<font color="{sc}"><b>[{sev}]</b></font>  '
                f'<b>{name}</b>  --  {l1}. {l2}',
                styles["body"],
            ))
            story.append(Spacer(1, 1.5 * mm))

    if summary:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(summary, styles["body"]))

    # ── Disclaimer ──────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(
        width="100%", thickness=0.4,
        color=colors.HexColor("#E2E8F0"),
        spaceAfter=2 * mm,
    ))
    story.append(Paragraph(
        "Research demonstration only -- not for clinical use. "
        "Peak width is a signal-processing proxy, not a clinical measurement. "
        "This report does not constitute medical advice.",
        styles["disclaimer"],
    ))

    # ── Build ───────────────────────────────────────────────────────
    doc.build(story)
    return buf.getvalue()
