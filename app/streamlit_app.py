"""
streamlit_app.py -- ECG AI Analyzer  |  Interactive Web Application

Professional Streamlit interface wrapping the existing ECG analysis
pipeline.  Provides a step-by-step guided workflow with interactive
Plotly visualizations, sidebar metrics dashboard, abnormality
highlighting, and PDF report export.

Launch:
    streamlit run streamlit_app.py

Requires:
    - All existing pipeline modules in app/
    - MIT-BIH data in data/mitdb/ (for sample records)
    - pip install streamlit plotly reportlab kaleido
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from bisect import bisect_left, bisect_right
from plotly.subplots import make_subplots

# ── Path setup ──────────────────────────────────────────────────────
APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
sys.path.insert(0, APP_DIR)

from preprocess import preprocess_ecg
from peaks import detect_r_peaks_adaptive
from features import extract_features, ECGFeatures
from rules import apply_rules, RuleResults
from evaluate import evaluate_detection, EvalMetrics
from report import generate_pdf_report


# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG AI Analyzer",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Theme / CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Dark clinical theme ────────────────────────────────── */
    .stApp {
        background-color: #0B1120;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F1923;
        border-right: 1px solid #1E3A50;
    }

    /* ── Cards ───────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #131D2B 0%, #182633 100%);
        border: 1px solid #1E3A50;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .metric-card .label {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
        font-family: 'Courier New', monospace;
    }
    .metric-card .value {
        font-size: 24px;
        font-weight: 700;
        color: #E2E8F0;
        font-family: 'Courier New', monospace;
    }
    .metric-card .unit {
        font-size: 12px;
        color: #64748B;
        margin-left: 4px;
    }

    /* ── Status badges ──────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        letter-spacing: 0.5px;
    }
    .badge-normal   { background: #22C55E22; color: #22C55E; border: 1px solid #22C55E44; }
    .badge-mild     { background: #FBBF2422; color: #FBBF24; border: 1px solid #FBBF2444; }
    .badge-moderate { background: #F9731622; color: #F97316; border: 1px solid #F9731644; }
    .badge-significant { background: #EF444422; color: #EF4444; border: 1px solid #EF444444; }
    .badge-excellent { background: #22C55E22; color: #22C55E; border: 1px solid #22C55E44; }
    .badge-good     { background: #FBBF2422; color: #FBBF24; border: 1px solid #FBBF2444; }
    .badge-fair     { background: #F9731622; color: #F97316; border: 1px solid #F9731644; }

    /* ── Finding rows ───────────────────────────────────────── */
    .finding-row {
        background: #131D2B;
        border: 1px solid #1E3A50;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .finding-title {
        font-size: 13px;
        font-weight: 700;
        color: #E2E8F0;
        font-family: 'Courier New', monospace;
    }
    .finding-detail {
        font-size: 11px;
        color: #94A3B8;
        font-family: 'Courier New', monospace;
        margin-top: 4px;
    }

    /* ── Step header ────────────────────────────────────────── */
    .step-header {
        font-size: 15px;
        font-weight: 700;
        color: #00D4AA;
        font-family: 'Courier New', monospace;
        padding: 10px 0 6px 0;
        border-bottom: 1px solid #1E3A50;
        margin-bottom: 14px;
    }

    /* ── Section divider ────────────────────────────────────── */
    .sidebar-section {
        font-size: 11px;
        font-weight: 700;
        color: #00D4AA;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Courier New', monospace;
        margin-top: 18px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #1E3A5066;
    }

    /* ── Disclaimer ─────────────────────────────────────────── */
    .disclaimer {
        font-size: 10px;
        color: #475569;
        font-style: italic;
        text-align: center;
        padding: 12px;
        border-top: 1px solid #1E3A5044;
        margin-top: 20px;
        font-family: 'Courier New', monospace;
    }

    /* ── Hide streamlit defaults for cleaner look ───────────── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Plotly chart container ──────────────────────────────── */
    .stPlotlyChart {
        border: 1px solid #1E3A50;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Integrated nav module ──────────────────────────────── */
    .nav-module {
        background: #0F1923;
        border: 1px solid #1E3A50;
        border-radius: 10px;
        padding: 12px 16px 6px 16px;
        margin: 10px 0 14px 0;
    }
    .nav-module .stPlotlyChart {
        border: none !important;
        border-radius: 0 !important;
    }
    .nav-module-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .nav-module-title {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Courier New', monospace;
    }
    .nav-viewing-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        font-family: 'Courier New', monospace;
        letter-spacing: 0.3px;
        animation: pulse-badge 2s ease-in-out infinite;
    }
    .nav-viewing-badge.active {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .nav-viewing-badge.normal {
        background: rgba(34,197,94,0.1);
        color: #22C55E;
        border: 1px solid rgba(34,197,94,0.2);
    }
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .nav-btn-row {
        display: flex;
        gap: 6px;
        align-items: center;
        margin-top: 2px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Plotly theme ────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F1923",
    plot_bgcolor="#131D2B",
    font=dict(family="Courier New, monospace", color="#94A3B8", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(30,58,80,0.33)", zerolinecolor="rgba(30,58,80,0.33)",
        title_font=dict(color="#64748B"),
    ),
    yaxis=dict(
        gridcolor="rgba(30,58,80,0.33)", zerolinecolor="rgba(30,58,80,0.33)",
        title_font=dict(color="#64748B"),
    ),
    legend=dict(
        bgcolor="rgba(15,25,35,0)", bordercolor="#1E3A50",
        font=dict(size=10, color="#94A3B8"),
    ),
    dragmode="zoom",
    hovermode="x unified",
)

C_SIGNAL   = "#00D4AA"
C_PEAK     = "#EF4444"
C_ANNOT    = "#FBBF24"
C_ABNORMAL = "rgba(239, 68, 68, 0.12)"


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _metric_card(label: str, value: str, unit: str = "") -> str:
    """Return HTML for a single metric card."""
    unit_span = f'<span class="unit">{unit}</span>' if unit else ""
    return (
        f'<div class="metric-card">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{value}{unit_span}</div>'
        f'</div>'
    )


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{text}</span>'


def _make_short_finding(flag) -> dict:
    """Convert a rule flag into short dashboard-safe text."""
    name = flag.name.replace("_", " ").title()
    sev = flag.severity.upper()
    if flag.name == "tachycardia":
        l1, l2 = f"Mean HR {flag.value:.0f} BPM", f"Thr: {flag.threshold:.0f} BPM"
    elif flag.name == "bradycardia":
        l1, l2 = f"Mean HR {flag.value:.0f} BPM", f"Thr: {flag.threshold:.0f} BPM"
    elif flag.name == "premature_beats":
        l1, l2 = f"{flag.value:.0f} premature beats", "Possible PAC/PVC"
    elif flag.name == "irregular_rhythm":
        l1, l2 = f"SDNN {flag.value*1000:.0f} ms", "Elevated RR variability"
    elif flag.name == "wide_qrs":
        l1, l2 = f"Width {flag.value:.0f} ms (proxy)", f"Thr: {flag.threshold:.0f} ms"
    else:
        l1, l2 = f"Value: {flag.value:.2f}", f"Thr: {flag.threshold:.2f}"
    return {"severity": sev, "title": name, "line1": l1, "line2": l2}


def _build_ecg_figure(
    signal, fs, title="ECG Signal",
    detected_peaks=None, annotation_peaks=None,
    start_sec=0.0, duration_sec=10.0,
    highlight_abnormal=False, feats=None,
    height=420,
    cached_segments=None,
):
    """Build a Plotly figure for the ECG signal."""
    s0 = max(0, int(start_sec * fs))
    s1 = min(len(signal), int((start_sec + duration_sec) * fs))
    t = np.arange(s0, s1) / fs
    seg = signal[s0:s1]

    fig = go.Figure()

    # Abnormal segment highlighting — viewport-filtered, batched shapes
    if highlight_abnormal:
        t_start = s0 / fs
        t_end = s1 / fs
        if cached_segments is not None and len(cached_segments) > 0:
            shapes = []
            for seg_s, seg_e in cached_segments:
                if seg_s >= t_end:
                    break
                if seg_e > t_start:
                    shapes.append(dict(
                        type="rect", xref="x", yref="paper",
                        x0=seg_s, x1=seg_e, y0=0, y1=1,
                        fillcolor=C_ABNORMAL, line_width=0,
                        layer="below",
                    ))
            if shapes:
                fig.update_layout(shapes=shapes)
        elif feats is not None and detected_peaks is not None:
            ihr = feats.instantaneous_hr
            if len(detected_peaks) > 1 and len(ihr) > 0:
                shapes = []
                for i in range(len(ihr)):
                    pk_s = detected_peaks[i]
                    pk_e = detected_peaks[i+1] if (i+1) < len(detected_peaks) else pk_s
                    if pk_e / fs <= t_start:
                        continue
                    if pk_s / fs >= t_end:
                        break
                    if ihr[i] > 100 or ihr[i] < 60:
                        shapes.append(dict(
                            type="rect", xref="x", yref="paper",
                            x0=pk_s/fs, x1=pk_e/fs, y0=0, y1=1,
                            fillcolor=C_ABNORMAL, line_width=0,
                            layer="below",
                        ))
                if shapes:
                    fig.update_layout(shapes=shapes)

    # ECG trace
    fig.add_trace(go.Scattergl(
        x=t, y=seg, mode="lines",
        line=dict(color=C_SIGNAL, width=1.2),
        name="ECG", hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}<extra></extra>",
    ))

    # Detected peaks
    if detected_peaks is not None:
        mask = (detected_peaks >= s0) & (detected_peaks < s1)
        dp = detected_peaks[mask]
        if len(dp) > 0:
            fig.add_trace(go.Scattergl(
                x=dp / fs, y=signal[dp], mode="markers",
                marker=dict(color=C_PEAK, size=7, symbol="circle",
                            line=dict(color="white", width=0.8)),
                name="R-peaks",
                hovertemplate="R-peak at %{x:.3f}s<extra></extra>",
            ))

    # Expert annotations
    if annotation_peaks is not None:
        mask = (annotation_peaks >= s0) & (annotation_peaks < s1)
        ap = annotation_peaks[mask]
        if len(ap) > 0:
            fig.add_trace(go.Scattergl(
                x=ap / fs, y=signal[ap], mode="markers",
                marker=dict(color=C_ANNOT, size=5, symbol="diamond",
                            line=dict(color="white", width=0.4)),
                name="Expert", opacity=0.8,
                hovertemplate="Expert at %{x:.3f}s<extra></extra>",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#E2E8F0")),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=height,
    )
    return fig


def _load_mitbih(record_id: str):
    """Load a MIT-BIH record using the existing data_loader."""
    from data_loader import load_record, filter_beat_annotations
    ecg = load_record(record_id)
    ecg = filter_beat_annotations(ecg)
    return ecg.signal.copy(), ecg.fs, ecg.ann_indices, ecg.duration_sec


def _check_mitbih_available() -> list:
    """Return list of available MIT-BIH record IDs."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data", "mitdb")
    if not os.path.isdir(data_dir):
        return []
    records = sorted(set(
        f.replace(".hea", "")
        for f in os.listdir(data_dir)
        if f.endswith(".hea")
    ))
    return records


def _fig_to_png_bytes(fig, width=1400, height=400) -> bytes:
    """Export a Plotly figure to PNG bytes for the PDF report."""
    try:
        return fig.to_image(format="png", width=width, height=height,
                            scale=2, engine="kaleido")
    except Exception:
        # kaleido not available -- skip image in report
        return None


def _build_abnormal_minimap(detected_peaks, feats, fs, duration_sec,
                            viewport_start=None, viewport_dur=10.0,
                            cached_segments=None):
    """Build a thin Plotly timeline bar showing abnormal segment locations."""
    fig = go.Figure()

    shapes = [dict(
        type="rect", x0=0, x1=duration_sec, y0=0, y1=1,
        fillcolor="#131D2B", line=dict(color="#1E3A50", width=1),
    )]

    if cached_segments is not None:
        for seg_s, seg_e in cached_segments:
            shapes.append(dict(
                type="rect",
                x0=seg_s, x1=seg_e, y0=0, y1=1,
                fillcolor="rgba(239,68,68,0.55)", line_width=0,
            ))
    elif feats is not None and detected_peaks is not None:
        ihr = feats.instantaneous_hr
        if len(detected_peaks) > 1 and len(ihr) > 0:
            for i in range(len(ihr)):
                pk_s = detected_peaks[i]
                pk_e = detected_peaks[i + 1] if (i + 1) < len(detected_peaks) else pk_s
                if ihr[i] > 100 or ihr[i] < 60:
                    shapes.append(dict(
                        type="rect",
                        x0=pk_s / fs, x1=pk_e / fs, y0=0, y1=1,
                        fillcolor="rgba(239,68,68,0.55)", line_width=0,
                    ))

    if viewport_start is not None:
        shapes.append(dict(
            type="rect",
            x0=viewport_start, x1=viewport_start + viewport_dur,
            y0=0, y1=1,
            fillcolor="rgba(0,212,170,0.12)",
            line=dict(color="rgba(0,212,170,0.6)", width=1.5),
        ))

    fig.add_trace(go.Scatter(
        x=[0, duration_sec], y=[0.5, 0.5],
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        hoverinfo="x", showlegend=False,
    ))

    fig.update_layout(
        shapes=shapes,
        height=36, margin=dict(l=50, r=20, t=0, b=0),
        paper_bgcolor="#0F1923", plot_bgcolor="#131D2B",
        xaxis=dict(
            range=[0, duration_sec], showgrid=False,
            zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            range=[0, 1], showgrid=False, zeroline=False,
            showticklabels=False, fixedrange=True,
        ),
        dragmode=False, hovermode="x",
    )
    return fig


def _get_abnormal_times(detected_peaks, feats, fs):
    """Extract start times (seconds) of abnormal segments from existing data."""
    times = []
    if feats is None or detected_peaks is None:
        return times
    ihr = feats.instantaneous_hr
    if len(detected_peaks) < 2 or len(ihr) == 0:
        return times
    for i in range(len(ihr)):
        if ihr[i] > 100 or ihr[i] < 60:
            t = detected_peaks[i] / fs
            if len(times) == 0 or (t - times[-1]) > 2.0:
                times.append(t)
    return times


def _is_viewing_abnormal(win_start, win_dur, detected_peaks, feats, fs):
    """Check if the current viewport overlaps any abnormal segment."""
    if feats is None or detected_peaks is None:
        return False
    ihr = feats.instantaneous_hr
    if len(detected_peaks) < 2 or len(ihr) == 0:
        return False
    win_end = win_start + win_dur
    for i in range(len(ihr)):
        if ihr[i] > 100 or ihr[i] < 60:
            seg_s = detected_peaks[i] / fs
            seg_e = (detected_peaks[i + 1] / fs
                     if (i + 1) < len(detected_peaks)
                     else seg_s)
            if seg_s < win_end and seg_e > win_start:
                return True
    return False


def _precompute_abnormal_data(detected_peaks, feats, fs):
    """Precompute all abnormality navigation data once.

    Returns a dict with:
        segments        : list of (start_sec, end_sec) for every abnormal interval
        merged_segments : adjacent segments merged to reduce draw calls
        seg_starts      : sorted array of segment start times for bisect
        nav_times       : deduplicated list of navigation target times
        nav_arr         : numpy array of nav_times for fast argmin
    """
    segments = []
    nav_times = []
    empty = {"segments": [], "merged_segments": [], "seg_starts": np.array([]),
             "nav_times": [], "nav_arr": np.array([])}
    if feats is None or detected_peaks is None:
        return empty
    ihr = feats.instantaneous_hr
    if len(detected_peaks) < 2 or len(ihr) == 0:
        return empty

    for i in range(len(ihr)):
        if ihr[i] > 100 or ihr[i] < 60:
            seg_s = detected_peaks[i] / fs
            seg_e = (detected_peaks[i + 1] / fs
                     if (i + 1) < len(detected_peaks) else seg_s)
            segments.append((seg_s, seg_e))
            if len(nav_times) == 0 or (seg_s - nav_times[-1]) > 2.0:
                nav_times.append(seg_s)

    # Merge adjacent/overlapping segments (gap < 0.05s)
    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    seg_starts = np.array([s for s, _ in merged]) if merged else np.array([])

    return {
        "segments": segments,
        "merged_segments": merged,
        "seg_starts": seg_starts,
        "nav_times": nav_times,
        "nav_arr": np.array(nav_times) if nav_times else np.array([]),
    }


def _is_viewing_abnormal_fast(win_start, win_dur, cached_segments):
    """Fast viewport-overlap check using precomputed segments."""
    win_end = win_start + win_dur
    for seg_s, seg_e in cached_segments:
        if seg_s >= win_end:
            break
        if seg_e > win_start:
            return True
    return False


# ════════════════════════════════════════════════════════════════════
# ABNORMALITY NAVIGATION FRAGMENT
# ════════════════════════════════════════════════════════════════════

def _nav_body():
    """Abnormality navigation — runs as isolated fragment when possible."""
    if st.session_state.cached_abn is None:
        st.session_state.cached_abn = _precompute_abnormal_data(
            st.session_state.detected_peaks,
            st.session_state.feats,
            st.session_state.fs,
        )
    _cab = st.session_state.cached_abn
    _abn_times = _cab["nav_times"]
    _abn_segs = _cab["merged_segments"]
    _abn_arr = _cab["nav_arr"]
    _max_start = max(0.0, st.session_state.duration_sec - 10.0)
    _n_abn = len(_abn_times)

    if "ab_nav_pos" not in st.session_state:
        st.session_state.ab_nav_pos = 0.0
    if "ab_nav_idx" not in st.session_state:
        st.session_state.ab_nav_idx = 0
    if _n_abn > 0:
        st.session_state.ab_nav_idx = max(0, min(st.session_state.ab_nav_idx, _n_abn - 1))

    nav_btn_cols = st.columns([1, 1, 3])
    with nav_btn_cols[0]:
        if st.button("⏮ Prev", key="btn_prev_ab", use_container_width=True,
                     disabled=(_n_abn == 0)):
            st.session_state.ab_nav_idx = max(st.session_state.ab_nav_idx - 1, 0)
            st.session_state.ab_nav_pos = min(
                max(_abn_times[st.session_state.ab_nav_idx] - 2.0, 0.0), _max_start)
            st.rerun()
    with nav_btn_cols[1]:
        if st.button("Next ⏭", key="btn_next_ab", use_container_width=True,
                     disabled=(_n_abn == 0)):
            st.session_state.ab_nav_idx = min(st.session_state.ab_nav_idx + 1, _n_abn - 1)
            st.session_state.ab_nav_pos = min(
                max(_abn_times[st.session_state.ab_nav_idx] - 2.0, 0.0), _max_start)
            st.rerun()
    with nav_btn_cols[2]:
        _viewing_ab = _is_viewing_abnormal_fast(
            st.session_state.ab_nav_pos, 10.0, _abn_segs,
        )
        if _viewing_ab and _n_abn > 0:
            st.markdown(
                '<div style="text-align:right; padding-top:4px;">'
                '<span class="nav-viewing-badge active">'
                'VIEWING ABNORMAL SEGMENT</span>'
                f'<div style="font-size:11px;color:#94A3B8;'
                f'font-family:Courier New,monospace;text-align:right;'
                f'margin-top:2px;">'
                f'{st.session_state.ab_nav_idx + 1} / {_n_abn}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif _n_abn > 0:
            st.markdown(
                '<div style="text-align:right; padding-top:4px;">'
                f'<span class="nav-viewing-badge normal">'
                f'{_n_abn} abnormal region(s)</span>'
                f'<div style="font-size:11px;color:#64748B;'
                f'font-family:Courier New,monospace;text-align:right;'
                f'margin-top:2px;">'
                f'{st.session_state.ab_nav_idx + 1} / {_n_abn}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:right; padding-top:6px;">'
                '<span class="nav-viewing-badge normal">'
                'No abnormal regions</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="nav-module">', unsafe_allow_html=True)

    st.markdown(
        '<div class="nav-module-header">'
        '<span class="nav-module-title">SIGNAL TIMELINE  |  '
        'red = abnormal  |  teal = current view</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _mm_key = f"{st.session_state.ab_nav_pos:.1f}"
    if st.session_state.get("_mm_cache_key") != _mm_key:
        st.session_state._mm_cache = _build_abnormal_minimap(
            st.session_state.detected_peaks,
            st.session_state.feats,
            st.session_state.fs,
            st.session_state.duration_sec,
            viewport_start=st.session_state.ab_nav_pos,
            viewport_dur=10.0,
            cached_segments=_abn_segs,
        )
        st.session_state._mm_cache_key = _mm_key

    st.plotly_chart(st.session_state._mm_cache, use_container_width=True,
                    config={"displayModeBar": False})

    win_start_ab = st.slider(
        "Navigate", 0.0, _max_start,
        st.session_state.ab_nav_pos, 0.5,
        key="nav_abnormal",
    )
    st.session_state.ab_nav_pos = win_start_ab

    if _n_abn > 0:
        st.session_state.ab_nav_idx = int(np.argmin(np.abs(_abn_arr - win_start_ab)))

    st.markdown('</div>', unsafe_allow_html=True)

    _fig_key = f"{win_start_ab:.1f}"
    if st.session_state.get("_ecg_fig_key") != _fig_key:
        st.session_state._ecg_fig_cache = _build_ecg_figure(
            st.session_state.processed, st.session_state.fs,
            title="ECG -- Abnormal Segments Highlighted",
            detected_peaks=st.session_state.detected_peaks,
            annotation_peaks=st.session_state.ann_indices,
            start_sec=win_start_ab, duration_sec=10,
            highlight_abnormal=True, feats=st.session_state.feats,
            height=440,
            cached_segments=_abn_segs,
        )
        st.session_state._ecg_fig_key = _fig_key

    st.plotly_chart(st.session_state._ecg_fig_cache, use_container_width=True)


_render_nav = st.fragment(_nav_body) if hasattr(st, "fragment") else _nav_body

def _init_state():
    defaults = {
        "step": 0,
        "raw_signal": None,
        "fs": 360.0,
        "ann_indices": None,
        "processed": None,
        "detected_peaks": None,
        "feats": None,
        "rule_results": None,
        "eval_metrics": None,
        "record_id": "",
        "duration_sec": 0.0,
        "cached_abn": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="font-size:22px; font-weight:700; color:#00D4AA; '
        'font-family: Courier New, monospace; padding-bottom:4px;">'
        '💓 ECG AI Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:10px; color:#64748B; '
        'font-family: Courier New, monospace; margin-bottom:16px;">'
        'Interactive ECG Analysis Platform</div>',
        unsafe_allow_html=True,
    )

    # ── Data source ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">DATA SOURCE</div>',
                unsafe_allow_html=True)

    source = st.radio(
        "Input method",
        ["MIT-BIH Sample Record", "Upload CSV / NPY"],
        label_visibility="collapsed",
    )

    if source == "MIT-BIH Sample Record":
        available = _check_mitbih_available()
        if available:
            record_id = st.selectbox("Record ID", available, index=0)
        else:
            record_id = st.text_input("Record ID", value="100")
            st.caption(
                "Data not found. Download with:\n"
                '`python -c "import wfdb; '
                "wfdb.dl_database('mitdb', 'data/mitdb')\"`"
            )
    else:
        uploaded = st.file_uploader(
            "Upload ECG file", type=["csv", "npy"],
            help="CSV: single-column amplitude values. NPY: 1-D array.",
        )
        custom_fs = st.number_input("Sampling Frequency (Hz)", value=360.0,
                                     min_value=50.0, max_value=5000.0, step=10.0)

    load_btn = st.button("Load Signal", use_container_width=True, type="primary")

    # ── Metrics panel (filled dynamically) ──────────────────────
    if st.session_state.feats is not None:
        feats = st.session_state.feats

        st.markdown('<div class="sidebar-section">HEART METRICS</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(_metric_card("Mean HR", f"{feats.mean_hr:.0f}", "BPM"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card("Median HR", f"{feats.median_hr:.0f}", "BPM"),
                        unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(_metric_card("SDNN", f"{feats.sdnn*1000:.1f}", "ms"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(_metric_card("RMSSD", f"{feats.rmssd*1000:.1f}", "ms"),
                        unsafe_allow_html=True)

        st.markdown(_metric_card("pNN50", f"{feats.pnn50:.1f}", "%"),
                    unsafe_allow_html=True)

        pw_mean = float(np.mean(feats.qrs_duration_ms)) if (
            feats.qrs_duration_ms is not None and len(feats.qrs_duration_ms) > 0
        ) else 0.0
        st.markdown(
            _metric_card("Peak Width", f"{pw_mean:.1f} +/- {feats.peak_width_std_ms:.1f}", "ms"),
            unsafe_allow_html=True,
        )
        st.caption("*Half-prominence proxy, not clinical QRS*")

    # ── Detection performance ───────────────────────────────────
    if st.session_state.eval_metrics is not None:
        em = st.session_state.eval_metrics

        st.markdown('<div class="sidebar-section">DETECTION PERFORMANCE</div>',
                    unsafe_allow_html=True)

        cols = st.columns(3)
        for col, (lbl, val) in zip(cols, [
            ("Prec", f"{em.precision:.3f}"),
            ("Rec", f"{em.recall:.3f}"),
            ("F1", f"{em.f1:.3f}"),
        ]):
            with col:
                st.markdown(_metric_card(lbl, val), unsafe_allow_html=True)

        # F1 badge
        if em.f1 >= 0.95:
            st.markdown(_badge("EXCELLENT", "excellent"), unsafe_allow_html=True)
        elif em.f1 >= 0.90:
            st.markdown(_badge("GOOD", "good"), unsafe_allow_html=True)
        elif em.f1 >= 0.80:
            st.markdown(_badge("FAIR", "fair"), unsafe_allow_html=True)
        else:
            st.markdown(_badge("NEEDS WORK", "significant"), unsafe_allow_html=True)

    # ── Clinical findings ───────────────────────────────────────
    if st.session_state.rule_results is not None:
        rr = st.session_state.rule_results

        st.markdown('<div class="sidebar-section">CLINICAL FINDINGS</div>',
                    unsafe_allow_html=True)

        if rr.is_normal:
            st.markdown(_badge("NORMAL", "normal"), unsafe_allow_html=True)
            st.caption("No abnormalities detected.")
        else:
            for flag in rr.flags[:5]:
                info = _make_short_finding(flag)
                sev_cls = info["severity"].lower()
                st.markdown(
                    f'<div class="finding-row">'
                    f'  {_badge(info["severity"], sev_cls)} '
                    f'  <span class="finding-title">{info["title"]}</span>'
                    f'  <div class="finding-detail">{info["line1"]}  |  {info["line2"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Disclaimer ──────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        'Research demo only. Not for clinical use.'
        '</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════

# ── Title ───────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; padding: 10px 0 20px 0;">'
    '<span style="font-size:32px; font-weight:800; color:#00D4AA; '
    'font-family: Courier New, monospace;">ECG AI Analyzer</span>'
    '<br>'
    '<span style="font-size:13px; color:#475569; '
    'font-family: Courier New, monospace;">'
    'MIT-BIH Arrhythmia Database  |  Interactive Signal Processing Pipeline'
    '</span></div>',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD SIGNAL
# ════════════════════════════════════════════════════════════════════

if load_btn:
    try:
        if source == "MIT-BIH Sample Record":
            raw, fs, ann, dur = _load_mitbih(record_id)
            st.session_state.raw_signal = raw
            st.session_state.fs = fs
            st.session_state.ann_indices = ann
            st.session_state.record_id = record_id
            st.session_state.duration_sec = dur
        else:
            if uploaded is None:
                st.error("Please upload a file first.")
                st.stop()
            if uploaded.name.endswith(".npy"):
                raw = np.load(uploaded).astype(np.float64).flatten()
            else:
                raw = np.loadtxt(uploaded, delimiter=",").astype(np.float64).flatten()
            fs = custom_fs
            st.session_state.raw_signal = raw
            st.session_state.fs = fs
            st.session_state.ann_indices = None
            st.session_state.record_id = uploaded.name.split(".")[0]
            st.session_state.duration_sec = len(raw) / fs

        # Reset downstream state
        st.session_state.step = 1
        st.session_state.processed = None
        st.session_state.detected_peaks = None
        st.session_state.feats = None
        st.session_state.rule_results = None
        st.session_state.eval_metrics = None
        st.session_state.cached_abn = None

    except Exception as e:
        st.error(f"Failed to load signal: {e}")
        st.stop()


# ── Show raw signal ─────────────────────────────────────────────────
if st.session_state.step >= 1 and st.session_state.raw_signal is not None:
    st.markdown('<div class="step-header">STEP 1  |  Raw Signal Loaded</div>',
                unsafe_allow_html=True)

    raw = st.session_state.raw_signal
    fs = st.session_state.fs
    dur = st.session_state.duration_sec

    col_info = st.columns(4)
    col_info[0].metric("Record", st.session_state.record_id)
    col_info[1].metric("Samples", f"{len(raw):,}")
    col_info[2].metric("Fs", f"{fs:.0f} Hz")
    col_info[3].metric("Duration", f"{dur:.1f} s")

    # Window navigator
    max_start = max(0.0, dur - 10.0)
    win_start = st.slider(
        "Navigate signal (start time)",
        0.0, max_start, 0.0, 0.5,
        key="nav_raw",
        help="Drag to browse through the signal",
    )
    win_dur = st.select_slider(
        "Window duration (s)",
        options=[5, 8, 10, 15, 20, 30],
        value=10,
        key="dur_raw",
    )

    fig_raw = _build_ecg_figure(
        raw, fs, title="Raw ECG Signal",
        annotation_peaks=st.session_state.ann_indices,
        start_sec=win_start, duration_sec=win_dur,
    )
    st.plotly_chart(fig_raw, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS
# ════════════════════════════════════════════════════════════════════

if st.session_state.step >= 1 and st.session_state.raw_signal is not None:
    st.markdown('<div class="step-header">STEP 2  |  Preprocessing</div>',
                unsafe_allow_html=True)

    if st.session_state.processed is None:
        if st.button("⚡ Preprocess Signal", type="primary", key="btn_preprocess"):
            with st.spinner("Applying bandpass filter (0.5-40 Hz) + normalization..."):
                processed = preprocess_ecg(
                    st.session_state.raw_signal, st.session_state.fs,
                )
                st.session_state.processed = processed
                st.session_state.step = max(st.session_state.step, 2)
            st.rerun()
    else:
        st.success("Signal preprocessed (bandpass 0.5-40 Hz + z-normalization)")

        win_start_p = st.slider(
            "Navigate", 0.0,
            max(0.0, st.session_state.duration_sec - 10.0),
            0.0, 0.5, key="nav_proc",
        )
        fig_proc = _build_ecg_figure(
            st.session_state.processed, st.session_state.fs,
            title="Preprocessed ECG",
            annotation_peaks=st.session_state.ann_indices,
            start_sec=win_start_p, duration_sec=10,
        )
        st.plotly_chart(fig_proc, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# STEP 3 — DETECT R-PEAKS
# ════════════════════════════════════════════════════════════════════

if st.session_state.step >= 2 and st.session_state.processed is not None:
    st.markdown('<div class="step-header">STEP 3  |  R-Peak Detection</div>',
                unsafe_allow_html=True)

    if st.session_state.detected_peaks is None:
        if st.button("🔍 Detect R-Peaks", type="primary", key="btn_peaks"):
            with st.spinner("Running adaptive threshold detection..."):
                peaks = detect_r_peaks_adaptive(
                    st.session_state.processed, st.session_state.fs,
                )
                st.session_state.detected_peaks = peaks
                st.session_state.step = max(st.session_state.step, 3)

                # Auto-evaluate if annotations available
                if st.session_state.ann_indices is not None:
                    em = evaluate_detection(
                        peaks, st.session_state.ann_indices,
                        st.session_state.fs, tolerance_ms=150.0,
                    )
                    st.session_state.eval_metrics = em
            st.rerun()
    else:
        n_peaks = len(st.session_state.detected_peaks)
        st.success(f"Detected {n_peaks:,} R-peaks")

        if st.session_state.eval_metrics is not None:
            em = st.session_state.eval_metrics
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Precision", f"{em.precision:.4f}")
            ec2.metric("Recall", f"{em.recall:.4f}")
            ec3.metric("F1 Score", f"{em.f1:.4f}")

        win_start_pk = st.slider(
            "Navigate", 0.0,
            max(0.0, st.session_state.duration_sec - 10.0),
            0.0, 0.5, key="nav_peaks",
        )
        fig_peaks = _build_ecg_figure(
            st.session_state.processed, st.session_state.fs,
            title="R-Peak Detection",
            detected_peaks=st.session_state.detected_peaks,
            annotation_peaks=st.session_state.ann_indices,
            start_sec=win_start_pk, duration_sec=10,
        )
        st.plotly_chart(fig_peaks, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE FEATURES
# ════════════════════════════════════════════════════════════════════

if st.session_state.step >= 3 and st.session_state.detected_peaks is not None:
    st.markdown('<div class="step-header">STEP 4  |  Clinical Metrics</div>',
                unsafe_allow_html=True)

    if st.session_state.feats is None:
        if st.button("📊 Compute Metrics", type="primary", key="btn_feats"):
            with st.spinner("Computing HRV features..."):
                feats = extract_features(
                    st.session_state.processed,
                    st.session_state.detected_peaks,
                    st.session_state.fs,
                )
                st.session_state.feats = feats
                st.session_state.step = max(st.session_state.step, 4)
            st.rerun()
    else:
        feats = st.session_state.feats
        st.success("Clinical metrics computed")

        # Metric cards in main area
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean HR", f"{feats.mean_hr:.1f} BPM")
        m2.metric("SDNN", f"{feats.sdnn*1000:.1f} ms")
        m3.metric("RMSSD", f"{feats.rmssd*1000:.1f} ms")
        m4.metric("pNN50", f"{feats.pnn50:.1f} %")

        # RR tachogram
        if len(feats.rr_intervals) > 0 and st.session_state.detected_peaks is not None:
            t_rr = st.session_state.detected_peaks[:-1].astype(float) / st.session_state.fs
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(
                x=t_rr, y=feats.rr_intervals * 1000,
                mode="lines+markers",
                line=dict(color="#3B82F6", width=1.2),
                marker=dict(size=3, color="#3B82F6"),
                name="RR Interval",
            ))
            fig_rr.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="RR Interval Tachogram", font=dict(size=13, color="#E2E8F0")),
                xaxis_title="Time (s)",
                yaxis_title="RR Interval (ms)",
                height=280,
            )
            st.plotly_chart(fig_rr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# STEP 5 — CLINICAL ANALYSIS + ABNORMALITY HIGHLIGHT
# ════════════════════════════════════════════════════════════════════

if st.session_state.step >= 4 and st.session_state.feats is not None:
    st.markdown('<div class="step-header">STEP 5  |  Clinical Analysis</div>',
                unsafe_allow_html=True)

    if st.session_state.rule_results is None:
        if st.button("🩺 Analyze Rhythm", type="primary", key="btn_rules"):
            with st.spinner("Applying rule-based analysis..."):
                rr = apply_rules(st.session_state.feats)
                st.session_state.rule_results = rr
                st.session_state.step = max(st.session_state.step, 5)
                st.session_state.cached_abn = _precompute_abnormal_data(
                    st.session_state.detected_peaks,
                    st.session_state.feats,
                    st.session_state.fs,
                )
            st.rerun()
    else:
        rr = st.session_state.rule_results

        if rr.is_normal:
            st.markdown(
                '<div style="text-align:center; padding:16px;">'
                + _badge("NORMAL -- No Abnormalities Detected", "normal")
                + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"{len(rr.flags)} finding(s) detected")
            for flag in rr.flags:
                info = _make_short_finding(flag)
                sev_cls = info["severity"].lower()
                st.markdown(
                    f'<div class="finding-row">'
                    f'  {_badge(info["severity"], sev_cls)}'
                    f'  <span class="finding-title" style="margin-left:8px;">'
                    f'{info["title"]}</span>'
                    f'  <div class="finding-detail">'
                    f'{info["line1"]}  |  {info["line2"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Integrated Abnormality Navigation Module ───────────────
        _render_nav()


# ════════════════════════════════════════════════════════════════════
# PDF REPORT EXPORT
# ════════════════════════════════════════════════════════════════════

if st.session_state.step >= 5 and st.session_state.rule_results is not None:
    st.markdown('<div class="step-header">EXPORT  |  Generate Report</div>',
                unsafe_allow_html=True)

    if st.button("📄 Generate PDF Report", type="primary", key="btn_pdf"):
        with st.spinner("Building PDF..."):
            feats = st.session_state.feats
            rr = st.session_state.rule_results
            em = st.session_state.eval_metrics

            # Try to capture current ECG figure as image for the PDF
            ecg_png = None
            try:
                fig_export = _build_ecg_figure(
                    st.session_state.processed, st.session_state.fs,
                    title="ECG Signal Snapshot",
                    detected_peaks=st.session_state.detected_peaks,
                    annotation_peaks=st.session_state.ann_indices,
                    start_sec=0.0, duration_sec=min(15.0, st.session_state.duration_sec),
                    highlight_abnormal=True, feats=feats,
                    height=350,
                )
                ecg_png = _fig_to_png_bytes(fig_export)
            except Exception:
                pass

            pw_mean = float(np.mean(feats.qrs_duration_ms)) if (
                feats.qrs_duration_ms is not None and len(feats.qrs_duration_ms) > 0
            ) else 0.0

            findings_list = None
            if not rr.is_normal:
                findings_list = [_make_short_finding(f) for f in rr.flags]

            pdf_bytes = generate_pdf_report(
                record_id=st.session_state.record_id,
                fs=st.session_state.fs,
                n_samples=len(st.session_state.raw_signal),
                mean_hr=feats.mean_hr,
                median_hr=feats.median_hr,
                sdnn_ms=feats.sdnn * 1000,
                rmssd_ms=feats.rmssd * 1000,
                pnn50=feats.pnn50,
                peak_width_mean_ms=pw_mean,
                peak_width_std_ms=feats.peak_width_std_ms,
                n_detected=len(st.session_state.detected_peaks),
                precision=em.precision if em else None,
                recall=em.recall if em else None,
                f1=em.f1 if em else None,
                n_reference=len(st.session_state.ann_indices) if st.session_state.ann_indices is not None else None,
                is_normal=rr.is_normal,
                findings=findings_list,
                summary=rr.summary,
                ecg_image_bytes=ecg_png,
            )

            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"ecg_report_{st.session_state.record_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("Report generated successfully!")


# ── Empty state ─────────────────────────────────────────────────────
if st.session_state.step == 0:
    st.markdown(
        '<div style="text-align:center; padding:60px 0; color:#334155; '
        'font-family: Courier New, monospace;">'
        '<div style="font-size:48px; margin-bottom:12px;">💓</div>'
        '<div style="font-size:16px; font-weight:600; color:#475569;">'
        'Select a data source and click Load Signal to begin</div>'
        '<div style="font-size:12px; color:#334155; margin-top:8px;">'
        'MIT-BIH Arrhythmia Database records or upload your own ECG data'
        '</div></div>',
        unsafe_allow_html=True,
    )
