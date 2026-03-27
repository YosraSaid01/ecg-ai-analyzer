"""
visualize.py — ECG Visualization

Produces clean, publication-quality matplotlib plots for:
    • Raw vs. preprocessed signal comparison
    • Detected R-peaks overlaid on the ECG
    • Expert annotation positions (ground truth)
    • Highlighted abnormal segments (from rule flags)
    • RR-interval tachogram and instantaneous heart rate

All plotting functions accept standard NumPy arrays and metadata so
they remain decoupled from the data-loading and detection modules.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple

from features import ECGFeatures
from rules import RuleResults


# ── Style defaults ─────────────────────────────────────────────────────

_COLORS = {
    "signal": "#1f77b4",
    "raw": "#aec7e8",
    "peak": "#d62728",
    "annotation": "#2ca02c",
    "abnormal": "#ff7f0e",
    "hr_line": "#9467bd",
}

_FIGURE_DPI = 120


def _time_axis(n_samples: int, fs: float) -> np.ndarray:
    """Create a time axis in seconds."""
    return np.arange(n_samples) / fs


def _window_bounds(fs: float, n_samples: int,
                   start_sec: float, duration_sec: float):
    """Compute sample-index bounds for a time window.

    Returns
    -------
    s0, s1 : int
        Start and end sample indices (end is exclusive).
    t : np.ndarray
        Time axis covering only the window (efficient — no full-signal alloc).
    """
    s0 = max(0, int(start_sec * fs))
    s1 = min(n_samples, int((start_sec + duration_sec) * fs))
    t = np.arange(s0, s1) / fs
    return s0, s1, t


def _peaks_in_window(peak_indices: np.ndarray, s0: int, s1: int) -> np.ndarray:
    """Return peak indices that fall within [s0, s1)."""
    mask = (peak_indices >= s0) & (peak_indices < s1)
    return peak_indices[mask]


# ── Public plotting functions ──────────────────────────────────────────


def plot_raw_vs_processed(raw: np.ndarray, processed: np.ndarray,
                          fs: float,
                          start_sec: float = 0.0, duration_sec: float = 10.0,
                          title: str = "Raw vs Processed ECG",
                          save_path: str = None) -> plt.Figure:
    """Plot raw and preprocessed ECG signals stacked vertically.

    Parameters
    ----------
    raw : np.ndarray
        Original ECG signal.
    processed : np.ndarray
        Preprocessed (filtered + normalized) ECG signal.
    fs : float
        Sampling frequency in Hz.
    start_sec : float
        Start time of the viewing window in seconds.
    duration_sec : float
        Duration of the viewing window in seconds.
    title : str
        Figure title.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s0, s1, t = _window_bounds(fs, len(raw), start_sec, duration_sec)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), dpi=_FIGURE_DPI,
                             sharex=True)

    axes[0].plot(t, raw[s0:s1], color=_COLORS["raw"], linewidth=0.6)
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Raw Signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, processed[s0:s1], color=_COLORS["signal"],
                 linewidth=0.6)
    axes[1].set_ylabel("Amplitude (normalized)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Preprocessed Signal")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_peaks(signal: np.ndarray, fs: float,
               detected_peaks: np.ndarray,
               annotation_peaks: np.ndarray = None,
               start_sec: float = 0.0, duration_sec: float = 10.0,
               title: str = "R-Peak Detection",
               save_path: str = None) -> plt.Figure:
    """Plot ECG signal with detected and annotated R-peaks.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    fs : float
        Sampling frequency in Hz.
    detected_peaks : np.ndarray
        Sample indices of detected R-peaks.
    annotation_peaks : np.ndarray, optional
        Sample indices of expert-annotated beats (ground truth).
    start_sec : float
        Start of viewing window (seconds).
    duration_sec : float
        Length of viewing window (seconds).
    title : str
        Figure title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s0, s1, t = _window_bounds(fs, len(signal), start_sec, duration_sec)

    fig, ax = plt.subplots(figsize=(14, 4), dpi=_FIGURE_DPI)
    ax.plot(t, signal[s0:s1], color=_COLORS["signal"], linewidth=0.6,
            label="ECG")

    # Detected peaks in window.
    det_in = _peaks_in_window(detected_peaks, s0, s1)
    ax.plot(det_in / fs, signal[det_in], "v", color=_COLORS["peak"],
            markersize=8, label="Detected R-peaks")

    # Annotation peaks in window.
    if annotation_peaks is not None:
        ann_in = _peaks_in_window(annotation_peaks, s0, s1)
        ax.plot(ann_in / fs, signal[ann_in], "^", color=_COLORS["annotation"],
                markersize=7, alpha=0.7, label="Expert annotations")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_with_abnormal_segments(
        signal: np.ndarray, fs: float,
        detected_peaks: np.ndarray,
        feats: ECGFeatures,
        rule_results: RuleResults,
        annotation_peaks: np.ndarray = None,
        start_sec: float = 0.0, duration_sec: float = 10.0,
        title: str = "ECG Analysis — Abnormal Segments",
        save_path: str = None) -> plt.Figure:
    """Plot ECG with abnormal RR intervals highlighted.

    Segments where the instantaneous heart rate exceeds the tachycardia
    threshold or drops below the bradycardia threshold are shaded.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    fs : float
        Sampling frequency in Hz.
    detected_peaks : np.ndarray
        Detected R-peak indices.
    feats : ECGFeatures
        Computed features (needs instantaneous_hr).
    rule_results : RuleResults
        Output of ``rules.apply_rules()``.
    annotation_peaks : np.ndarray, optional
        Expert annotation indices.
    start_sec : float
        Viewing window start (seconds).
    duration_sec : float
        Viewing window length (seconds).
    title : str
        Figure title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s0, s1, t = _window_bounds(fs, len(signal), start_sec, duration_sec)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=_FIGURE_DPI)
    ax.plot(t, signal[s0:s1], color=_COLORS["signal"], linewidth=0.6,
            label="ECG")

    # Shade abnormal inter-beat intervals (only those in view).
    ihr = feats.instantaneous_hr
    if len(detected_peaks) > 1 and len(ihr) > 0:
        tachy_thr = 100.0
        brady_thr = 60.0

        for i in range(len(ihr)):
            pk_start = detected_peaks[i]
            pk_end = detected_peaks[i + 1] if (i + 1) < len(detected_peaks) else pk_start
            if pk_start < s0 or pk_end >= s1:
                continue

            if ihr[i] > tachy_thr:
                ax.axvspan(pk_start / fs, pk_end / fs, alpha=0.15,
                           color="#d62728", zorder=0)
            elif ihr[i] < brady_thr:
                ax.axvspan(pk_start / fs, pk_end / fs, alpha=0.15,
                           color="#1f77b4", zorder=0)

    # Detected peaks.
    det_in = _peaks_in_window(detected_peaks, s0, s1)
    ax.plot(det_in / fs, signal[det_in], "v", color=_COLORS["peak"],
            markersize=8, label="Detected R-peaks")

    # Annotation peaks.
    if annotation_peaks is not None:
        ann_in = _peaks_in_window(annotation_peaks, s0, s1)
        ax.plot(ann_in / fs, signal[ann_in], "^", color=_COLORS["annotation"],
                markersize=7, alpha=0.7, label="Expert annotations")

    # Legend patches for shaded regions.
    handles, labels = ax.get_legend_handles_labels()
    if not rule_results.is_normal:
        handles.append(mpatches.Patch(color="#d62728", alpha=0.15,
                                      label="Tachycardia zone"))
        handles.append(mpatches.Patch(color="#1f77b4", alpha=0.15,
                                      label="Bradycardia zone"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_heart_rate(feats: ECGFeatures, peak_indices: np.ndarray,
                    fs: float,
                    start_sec: float = 0.0, duration_sec: float = 10.0,
                    title: str = "Instantaneous Heart Rate",
                    save_path: str = None) -> plt.Figure:
    """Plot instantaneous heart rate (tachogram) over time.

    Parameters
    ----------
    feats : ECGFeatures
        Computed features.
    peak_indices : np.ndarray
        Detected R-peak sample indices.
    fs : float
        Sampling frequency in Hz.
    start_sec : float
        Start of viewing window (seconds).
    duration_sec : float
        Length of viewing window (seconds).
    title : str
        Figure title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ihr = feats.instantaneous_hr
    if len(ihr) == 0 or len(peak_indices) < 2:
        fig, ax = plt.subplots(figsize=(14, 3), dpi=_FIGURE_DPI)
        ax.text(0.5, 0.5, "Insufficient peaks for heart-rate plot",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Time of each RR interval = start peak position.
    t_peaks = peak_indices[:-1].astype(np.float64) / fs

    # Window to the requested time range.
    end_sec = start_sec + duration_sec
    mask = (t_peaks >= start_sec) & (t_peaks < end_sec)
    t_win = t_peaks[mask]
    ihr_win = ihr[mask]

    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=_FIGURE_DPI)
    ax.plot(t_win, ihr_win, color=_COLORS["hr_line"], linewidth=1.0,
            marker=".", markersize=3)

    # Reference bands.
    ax.axhline(100, color="#d62728", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Tachycardia (100 BPM)")
    ax.axhline(60, color="#1f77b4", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Bradycardia (60 BPM)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_rr_intervals(feats: ECGFeatures, peak_indices: np.ndarray,
                      fs: float,
                      start_sec: float = 0.0, duration_sec: float = 10.0,
                      title: str = "RR-Interval Tachogram",
                      save_path: str = None) -> plt.Figure:
    """Plot successive RR intervals (Poincaré-style tachogram).

    Parameters
    ----------
    feats : ECGFeatures
        Computed features.
    peak_indices : np.ndarray
        Detected R-peak sample indices.
    fs : float
        Sampling frequency in Hz.
    start_sec : float
        Start of viewing window (seconds).
    duration_sec : float
        Length of viewing window (seconds).
    title : str
        Figure title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rr = feats.rr_intervals
    if len(rr) == 0:
        fig, ax = plt.subplots(figsize=(14, 3), dpi=_FIGURE_DPI)
        ax.text(0.5, 0.5, "Insufficient data for RR plot",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    t_peaks = peak_indices[:-1].astype(np.float64) / fs

    # Window to the requested time range.
    end_sec = start_sec + duration_sec
    mask = (t_peaks >= start_sec) & (t_peaks < end_sec)
    t_win = t_peaks[mask]
    rr_win = rr[mask]

    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=_FIGURE_DPI)
    ax.plot(t_win, rr_win * 1000.0, color=_COLORS["signal"], linewidth=1.0,
            marker=".", markersize=3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RR Interval (ms)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def _make_short_finding(flag) -> dict:
    """Convert a rule flag into short, dashboard-safe text.

    Returns dict with keys: severity, title, line1, line2.
    All strings are pre-truncated to fit the column.
    """
    name = flag.name.replace("_", " ").title()
    sev = flag.severity.upper()

    # Build concise description lines instead of raw sentences.
    if flag.name == "tachycardia":
        l1 = f"Mean HR {flag.value:.0f} BPM"
        l2 = f"Threshold: {flag.threshold:.0f} BPM"
    elif flag.name == "bradycardia":
        l1 = f"Mean HR {flag.value:.0f} BPM"
        l2 = f"Threshold: {flag.threshold:.0f} BPM"
    elif flag.name == "premature_beats":
        # Extract count from description if available
        l1 = f"Value: {flag.value:.1f}"
        l2 = "Possible premature contractions"
    elif flag.name == "irregular_rhythm":
        l1 = f"SDNN {flag.value * 1000:.0f} ms"
        l2 = "Elevated RR variability"
    elif flag.name == "wide_qrs":
        l1 = f"Width {flag.value:.0f} ms (proxy)"
        l2 = f"Threshold: {flag.threshold:.0f} ms"
    else:
        l1 = f"Value: {flag.value:.2f}"
        l2 = f"Threshold: {flag.threshold:.2f}"

    return {"severity": sev, "title": name, "line1": l1, "line2": l2}


def plot_summary_dashboard(
        signal: np.ndarray, fs: float,
        detected_peaks: np.ndarray,
        annotation_peaks: np.ndarray,
        feats: ECGFeatures,
        rule_results: RuleResults,
        eval_metrics=None,
        record_id: str = "",
        start_sec: float = 0.0, duration_sec: float = 12.0,
        save_path: str = None) -> plt.Figure:
    """Generate a single professional dashboard figure.

    Layout
    ------
    TOP:     ECG signal with R-peaks and annotations (10-15 s window)
    BOTTOM:  Three card panels — HRV metrics, detection performance,
             clinical findings — with safe text boundaries.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    fs : float
        Sampling frequency in Hz.
    detected_peaks : np.ndarray
        Detected R-peak sample indices.
    annotation_peaks : np.ndarray
        Expert-annotated beat indices.
    feats : ECGFeatures
        Computed clinical features.
    rule_results : RuleResults
        Rule engine output.
    eval_metrics : EvalMetrics or None
        Detection evaluation metrics (optional).
    record_id : str
        Record identifier for the title.
    start_sec : float
        Start of ECG viewing window (seconds).
    duration_sec : float
        Duration of ECG viewing window (seconds).
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.patches import FancyBboxPatch

    # ── Color palette ───────────────────────────────────────────────
    BG        = "#0F1923"
    PANEL_BG  = "#182633"
    CARD_BG   = "#1B2D3D"
    ACCENT    = "#00D4AA"
    SIGNAL_C  = "#00D4AA"
    PEAK_C    = "#EF4444"
    ANNOT_C   = "#FBBF24"
    TEXT_MAIN = "#E2E8F0"
    TEXT_DIM  = "#94A3B8"
    GRID_C    = "#1E3A50"
    DIVIDER   = "#2D4A5E"
    GREEN     = "#22C55E"
    ORANGE    = "#F97316"

    FONT = "monospace"

    # ── Figure setup ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12), dpi=150, facecolor=BG)

    # Master grid: header gap + ECG + cards + footer
    gs_master = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.65],
        hspace=0.07, left=0.04, right=0.96,
        top=0.89, bottom=0.05)

    # ── HEADER ──────────────────────────────────────────────────────
    prefix = f"Record {record_id}" if record_id else "ECG"
    fig.text(0.04, 0.98, "ECG AI Analyzer",
             fontsize=24, fontweight="bold", color=ACCENT,
             fontfamily=FONT, va="top")
    fig.text(0.04, 0.948,
             f"MIT-BIH Arrhythmia Database  |  {prefix}  |  "
             f"Fs = {fs:.0f} Hz",
             fontsize=10, color=TEXT_DIM, fontfamily=FONT, va="top")

    # Accent rule
    fig.patches.append(plt.Rectangle(
        (0.04, 0.930), 0.92, 0.0012,
        transform=fig.transFigure, facecolor=ACCENT, alpha=0.4,
        clip_on=False))

    # ── TOP — ECG SIGNAL ────────────────────────────────────────────
    ax_ecg = fig.add_subplot(gs_master[0])
    ax_ecg.set_facecolor(PANEL_BG)

    s0, s1, t = _window_bounds(fs, len(signal), start_sec, duration_sec)

    ax_ecg.plot(t, signal[s0:s1], color=SIGNAL_C, linewidth=1.0,
                alpha=0.9, zorder=2)

    det_in = _peaks_in_window(detected_peaks, s0, s1)
    if len(det_in) > 0:
        ax_ecg.plot(det_in / fs, signal[det_in], "o",
                    color=PEAK_C, markersize=6.5,
                    markeredgecolor="white", markeredgewidth=0.7,
                    zorder=3,
                    label=f"Detected  ({len(detected_peaks):,} total)")

    if annotation_peaks is not None:
        ann_in = _peaks_in_window(annotation_peaks, s0, s1)
        if len(ann_in) > 0:
            ax_ecg.plot(ann_in / fs, signal[ann_in], "D",
                        color=ANNOT_C, markersize=4.5, alpha=0.85,
                        markeredgecolor="white", markeredgewidth=0.4,
                        zorder=3, label="Expert")

    ax_ecg.set_ylabel("Amplitude", fontsize=10, color=TEXT_MAIN,
                       fontfamily=FONT)
    ax_ecg.set_xlabel("Time (s)", fontsize=10, color=TEXT_MAIN,
                       fontfamily=FONT)
    ax_ecg.set_title(
        f"Preprocessed ECG  |  "
        f"{start_sec:.0f} - {start_sec + duration_sec:.0f} s",
        fontsize=12, fontweight="bold", color=TEXT_MAIN,
        fontfamily=FONT, loc="left", pad=8)
    ax_ecg.legend(loc="upper right", fontsize=8.5,
                  facecolor=PANEL_BG, edgecolor=DIVIDER,
                  labelcolor=TEXT_DIM, prop={"family": FONT, "size": 8.5})
    ax_ecg.tick_params(colors=TEXT_DIM, labelsize=8)
    ax_ecg.grid(True, alpha=0.2, color=GRID_C, linewidth=0.4)
    for sp in ax_ecg.spines.values():
        sp.set_color(DIVIDER)
        sp.set_linewidth(0.5)

    # ── BOTTOM — THREE CARD PANELS ──────────────────────────────────
    gs_cards = gs_master[1].subgridspec(1, 3, wspace=0.04)

    def _make_card(gs_slot):
        """Create an axes that acts as a card with dark background."""
        ax = fig.add_subplot(gs_slot)
        ax.set_facecolor(CARD_BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        # Card border
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(DIVIDER)
            sp.set_linewidth(0.8)
        return ax

    # ════════════════════════════════════════════════════════════════
    # CARD 1 — HEART RATE & HRV
    # ════════════════════════════════════════════════════════════════
    c1 = _make_card(gs_cards[0])

    # Section title with underline
    PAD_L = 0.08          # left padding inside card
    PAD_R = 0.92          # right edge for values
    ROW_H = 0.095         # vertical step per metric row
    TITLE_Y = 0.93
    FIRST_ROW = 0.80

    c1.text(PAD_L, TITLE_Y, "HEART RATE & HRV",
            fontsize=11.5, fontweight="bold", color=ACCENT,
            fontfamily=FONT, transform=c1.transAxes, va="top")
    c1.plot([PAD_L, PAD_R], [0.865, 0.865], color=DIVIDER,
            linewidth=0.6, transform=c1.transAxes)

    # Metric rows: (label, value)
    hrv_rows = [
        ("Mean HR",    f"{feats.mean_hr:.1f} BPM"),
        ("Median HR",  f"{feats.median_hr:.1f} BPM"),
        ("SDNN",       f"{feats.sdnn * 1000:.1f} ms"),
        ("RMSSD",      f"{feats.rmssd * 1000:.1f} ms"),
        ("pNN50",      f"{feats.pnn50:.1f} %"),
    ]

    # Peak width
    pw_mean = (float(np.mean(feats.qrs_duration_ms))
               if feats.qrs_duration_ms is not None
               and len(feats.qrs_duration_ms) > 0 else 0.0)
    hrv_rows.append(
        ("Peak Width", f"{pw_mean:.1f} +/- {feats.peak_width_std_ms:.1f} ms"))

    for i, (label, value) in enumerate(hrv_rows):
        y = FIRST_ROW - i * ROW_H
        c1.text(PAD_L, y, label, fontsize=9.5, color=TEXT_DIM,
                fontfamily=FONT, transform=c1.transAxes, va="top")
        c1.text(PAD_R, y, value, fontsize=9.5, fontweight="bold",
                color=TEXT_MAIN, fontfamily=FONT, ha="right",
                transform=c1.transAxes, va="top")

    # Footnote — safe at bottom of card
    c1.text(PAD_L, 0.06,
            "Peak Width = half-prominence proxy\n"
            "Not a clinical QRS measurement",
            fontsize=7, color=TEXT_DIM, fontfamily=FONT,
            transform=c1.transAxes, va="bottom",
            linespacing=1.6, style="italic")

    # ════════════════════════════════════════════════════════════════
    # CARD 2 — DETECTION PERFORMANCE
    # ════════════════════════════════════════════════════════════════
    c2 = _make_card(gs_cards[1])

    c2.text(PAD_L, TITLE_Y, "DETECTION PERFORMANCE",
            fontsize=11.5, fontweight="bold", color=ACCENT,
            fontfamily=FONT, transform=c2.transAxes, va="top")
    c2.plot([PAD_L, PAD_R], [0.865, 0.865], color=DIVIDER,
            linewidth=0.6, transform=c2.transAxes)

    perf_rows = [
        ("Detected",  f"{len(detected_peaks):,} beats"),
    ]
    if annotation_peaks is not None:
        perf_rows.append(
            ("Reference", f"{len(annotation_peaks):,} beats"))

    if eval_metrics is not None:
        perf_rows.extend([
            ("Precision", f"{eval_metrics.precision:.4f}"),
            ("Recall",    f"{eval_metrics.recall:.4f}"),
            ("F1 Score",  f"{eval_metrics.f1:.4f}"),
        ])

    for i, (label, value) in enumerate(perf_rows):
        y = FIRST_ROW - i * ROW_H
        c2.text(PAD_L, y, label, fontsize=9.5, color=TEXT_DIM,
                fontfamily=FONT, transform=c2.transAxes, va="top")
        c2.text(PAD_R, y, value, fontsize=9.5, fontweight="bold",
                color=TEXT_MAIN, fontfamily=FONT, ha="right",
                transform=c2.transAxes, va="top")

    # F1 status badge — centered at bottom of card
    if eval_metrics is not None:
        f1 = eval_metrics.f1
        if f1 >= 0.95:
            bc, bt = GREEN, "EXCELLENT"
        elif f1 >= 0.90:
            bc, bt = ANNOT_C, "GOOD"
        elif f1 >= 0.80:
            bc, bt = ORANGE, "FAIR"
        else:
            bc, bt = PEAK_C, "NEEDS WORK"

        badge_w, badge_h = 0.50, 0.065
        badge_x = 0.5 - badge_w / 2
        badge_y = 0.07
        c2.add_patch(FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle="round,pad=0.01",
            facecolor=bc, alpha=0.15,
            edgecolor=bc, linewidth=1.2,
            transform=c2.transAxes, clip_on=False))
        c2.text(0.5, badge_y + badge_h / 2, bt,
                fontsize=12, fontweight="bold", color=bc,
                fontfamily=FONT, ha="center", va="center",
                transform=c2.transAxes)
    else:
        c2.text(0.5, 0.10, "Eval metrics unavailable",
                fontsize=9, color=TEXT_DIM, fontfamily=FONT,
                ha="center", transform=c2.transAxes)

    # ════════════════════════════════════════════════════════════════
    # CARD 3 — CLINICAL FINDINGS
    # ════════════════════════════════════════════════════════════════
    c3 = _make_card(gs_cards[2])

    c3.text(PAD_L, TITLE_Y, "CLINICAL FINDINGS",
            fontsize=11.5, fontweight="bold", color=ACCENT,
            fontfamily=FONT, transform=c3.transAxes, va="top")
    c3.plot([PAD_L, PAD_R], [0.865, 0.865], color=DIVIDER,
            linewidth=0.6, transform=c3.transAxes)

    SEV_COLORS = {
        "INFO": TEXT_DIM, "MILD": ANNOT_C,
        "MODERATE": ORANGE, "SIGNIFICANT": PEAK_C,
    }

    if rule_results.is_normal:
        # Normal badge
        nb_w, nb_h = 0.70, 0.07
        nb_x = 0.5 - nb_w / 2
        nb_y = 0.68
        c3.add_patch(FancyBboxPatch(
            (nb_x, nb_y), nb_w, nb_h,
            boxstyle="round,pad=0.01",
            facecolor=GREEN, alpha=0.12,
            edgecolor=GREEN, linewidth=1.0,
            transform=c3.transAxes, clip_on=False))
        c3.text(0.5, nb_y + nb_h / 2, "NORMAL",
                fontsize=13, fontweight="bold", color=GREEN,
                fontfamily=FONT, ha="center", va="center",
                transform=c3.transAxes)

        c3.text(0.5, 0.58, "No abnormalities detected",
                fontsize=9.5, color=TEXT_MAIN, fontfamily=FONT,
                ha="center", transform=c3.transAxes, va="top")
        c3.text(0.5, 0.48,
                "Rhythm appears regular\nwithin normal limits.",
                fontsize=9, color=TEXT_DIM, fontfamily=FONT,
                ha="center", transform=c3.transAxes, va="top",
                linespacing=1.5)
    else:
        y_pos = FIRST_ROW
        max_flags = 4  # safe max before running out of space
        for flag in rule_results.flags[:max_flags]:
            info = _make_short_finding(flag)
            sev_c = SEV_COLORS.get(info["severity"], TEXT_DIM)

            # Severity pill
            c3.text(PAD_L, y_pos, info["severity"],
                    fontsize=8, fontweight="bold", color=sev_c,
                    fontfamily=FONT, transform=c3.transAxes, va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=sev_c, alpha=0.12,
                              edgecolor="none"))

            # Finding title
            c3.text(PAD_L + 0.22, y_pos, info["title"],
                    fontsize=9.5, fontweight="bold", color=TEXT_MAIN,
                    fontfamily=FONT, transform=c3.transAxes, va="top")

            # Detail lines
            c3.text(PAD_L + 0.02, y_pos - 0.065, info["line1"],
                    fontsize=8, color=TEXT_DIM, fontfamily=FONT,
                    transform=c3.transAxes, va="top")
            c3.text(PAD_L + 0.02, y_pos - 0.12, info["line2"],
                    fontsize=8, color=TEXT_DIM, fontfamily=FONT,
                    transform=c3.transAxes, va="top")

            y_pos -= 0.20  # safe spacing between flags

        remaining = len(rule_results.flags) - max_flags
        if remaining > 0:
            c3.text(PAD_L, y_pos,
                    f"+{remaining} more finding(s)",
                    fontsize=8, color=TEXT_DIM, fontfamily=FONT,
                    transform=c3.transAxes, va="top", style="italic")

    # ── DISCLAIMER FOOTER ───────────────────────────────────────────
    fig.text(0.5, 0.015,
             "Research demonstration only  --  not for clinical use.  "
             "Peak width is a signal-processing proxy, "
             "not a clinical measurement.",
             fontsize=7.5, color=TEXT_DIM, fontfamily=FONT,
             ha="center", va="bottom", style="italic")

    # ── SAVE ────────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=BG,
                    edgecolor="none", dpi=150)

    return fig


def show_all(signal_raw: np.ndarray, signal_processed: np.ndarray,
             fs: float,
             detected_peaks: np.ndarray,
             annotation_peaks: np.ndarray,
             feats: ECGFeatures,
             rule_results: RuleResults,
             record_id: str = "",
             start_sec: float = 0.0, duration_sec: float = 10.0,
             save_dir: str = None,
             eval_metrics=None) -> List[plt.Figure]:
    """Generate and display all standard plots.

    This is the single entry point called by main.py.

    Parameters
    ----------
    signal_raw : np.ndarray
        Original raw ECG signal.
    signal_processed : np.ndarray
        Preprocessed ECG signal.
    fs : float
        Sampling frequency in Hz.
    detected_peaks : np.ndarray
        Detected R-peak sample indices.
    annotation_peaks : np.ndarray
        Expert-annotated beat sample indices.
    feats : ECGFeatures
        Computed clinical features.
    rule_results : RuleResults
        Abnormality flags from the rule engine.
    record_id : str
        MIT-BIH record ID (for titles).
    start_sec : float
        Start of viewing window (seconds).
    duration_sec : float
        Length of viewing window (seconds).
    save_dir : str, optional
        If provided, save all figures as PNG files in this directory.
    eval_metrics : EvalMetrics or None, optional
        Detection evaluation metrics (for the summary dashboard).

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    import os

    prefix = f"Record {record_id}" if record_id else "ECG"

    save = lambda name: os.path.join(save_dir, name) if save_dir else None

    figs = []

    # ── PRIMARY: Professional summary dashboard ─────────────────────
    figs.append(plot_summary_dashboard(
        signal=signal_processed,
        fs=fs,
        detected_peaks=detected_peaks,
        annotation_peaks=annotation_peaks,
        feats=feats,
        rule_results=rule_results,
        eval_metrics=eval_metrics,
        record_id=record_id,
        start_sec=start_sec,
        duration_sec=duration_sec,
        save_path=save(f"record_{record_id}_summary.png"),
    ))

    # ── Legacy individual plots (still generated for detail) ────────
    figs.append(plot_raw_vs_processed(
        signal_raw, signal_processed, fs,
        start_sec=start_sec, duration_sec=duration_sec,
        title=f"{prefix} — Raw vs Processed",
        save_path=save(f"{record_id}_raw_vs_processed.png"),
    ))

    figs.append(plot_peaks(
        signal_processed, fs, detected_peaks,
        annotation_peaks=annotation_peaks,
        start_sec=start_sec, duration_sec=duration_sec,
        title=f"{prefix} — R-Peak Detection",
        save_path=save(f"{record_id}_peaks.png"),
    ))

    figs.append(plot_with_abnormal_segments(
        signal_processed, fs, detected_peaks, feats, rule_results,
        annotation_peaks=annotation_peaks,
        start_sec=start_sec, duration_sec=duration_sec,
        title=f"{prefix} — Abnormal Segments",
        save_path=save(f"{record_id}_abnormal.png"),
    ))

    figs.append(plot_heart_rate(
        feats, detected_peaks, fs,
        start_sec=start_sec, duration_sec=duration_sec,
        title=f"{prefix} — Instantaneous Heart Rate",
        save_path=save(f"{record_id}_heart_rate.png"),
    ))

    figs.append(plot_rr_intervals(
        feats, detected_peaks, fs,
        start_sec=start_sec, duration_sec=duration_sec,
        title=f"{prefix} — RR Intervals",
        save_path=save(f"{record_id}_rr_intervals.png"),
    ))

    plt.show()
    return figs


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("visualize.py — run via main.py for full plots.")
    print("Module imports OK.")
