"""
features.py — Clinical Feature Extraction

Computes clinically interpretable features from detected R-peak positions:
    • RR intervals (seconds)
    • Instantaneous heart rate (BPM per beat)
    • Mean / median heart rate
    • Heart-rate variability metrics (SDNN, RMSSD, pNN50)
    • Approximate peak width at half-prominence (proxy, not clinical QRS)

All functions operate on peak-index arrays and the sampling frequency,
making them independent of the detection method used upstream.

References:
    Task Force of the European Society of Cardiology. Heart rate
    variability — standards of measurement, physiological interpretation,
    and clinical use. Circulation. 1996;93(5):1043-1065.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import peak_widths


@dataclass
class ECGFeatures:
    """Container for features derived from a single ECG record.

    Attributes
    ----------
    rr_intervals : np.ndarray
        Successive RR intervals in seconds.
    instantaneous_hr : np.ndarray
        Beat-by-beat heart rate in BPM (length = len(rr_intervals)).
    mean_hr : float
        Mean heart rate in BPM.
    median_hr : float
        Median heart rate in BPM.
    sdnn : float
        Standard deviation of RR intervals (seconds).  Primary
        time-domain HRV metric.
    rmssd : float
        Root mean square of successive RR differences (seconds).
        Reflects short-term (parasympathetic) variability.
    pnn50 : float
        Percentage of successive RR intervals differing by > 50 ms.
    qrs_duration_ms : Optional[np.ndarray]
        Approximate peak width at half-prominence per beat (ms).
        This is a signal-processing proxy, NOT a clinical QRS
        delineation.  Kept as ``qrs_duration_ms`` for backward
        compatibility with the rule engine.
    peak_width_std_ms : float
        Standard deviation of peak widths across all beats (ms).
    """
    rr_intervals: np.ndarray
    instantaneous_hr: np.ndarray
    mean_hr: float
    median_hr: float
    sdnn: float
    rmssd: float
    pnn50: float
    qrs_duration_ms: Optional[np.ndarray] = None
    peak_width_std_ms: float = 0.0


# ── Core feature functions ─────────────────────────────────────────────


def compute_rr_intervals(peak_indices: np.ndarray, fs: float) -> np.ndarray:
    """Compute successive RR intervals in seconds.

    Parameters
    ----------
    peak_indices : np.ndarray
        Sorted sample indices of detected R-peaks.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        RR intervals in seconds (length = len(peak_indices) - 1).
    """
    if len(peak_indices) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(peak_indices).astype(np.float64) / fs


def compute_instantaneous_hr(rr_intervals: np.ndarray) -> np.ndarray:
    """Convert RR intervals to instantaneous heart rate in BPM.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    np.ndarray
        Heart rate in beats per minute for each interval.
    """
    # Guard against division by zero for any degenerate intervals.
    safe_rr = np.where(rr_intervals > 0, rr_intervals, np.nan)
    return 60.0 / safe_rr


def compute_sdnn(rr_intervals: np.ndarray) -> float:
    """Standard deviation of RR intervals (SDNN).

    SDNN reflects overall heart-rate variability and is influenced by
    both sympathetic and parasympathetic activity.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    float
        SDNN in seconds.
    """
    if len(rr_intervals) < 2:
        return 0.0
    return float(np.std(rr_intervals, ddof=1))


def compute_rmssd(rr_intervals: np.ndarray) -> float:
    """Root mean square of successive RR differences (RMSSD).

    RMSSD is a short-term HRV metric primarily reflecting
    parasympathetic (vagal) modulation.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    float
        RMSSD in seconds.
    """
    if len(rr_intervals) < 2:
        return 0.0
    diffs = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(diffs ** 2)))


def compute_pnn50(rr_intervals: np.ndarray) -> float:
    """Percentage of successive RR intervals differing by > 50 ms.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    float
        pNN50 as a percentage (0–100).
    """
    if len(rr_intervals) < 2:
        return 0.0
    diffs = np.abs(np.diff(rr_intervals))
    return float(100.0 * np.sum(diffs > 0.050) / len(diffs))


def estimate_peak_width(signal: np.ndarray, peak_indices: np.ndarray,
                        fs: float) -> np.ndarray:
    """Estimate approximate peak width at half-prominence for each R-peak.

    Uses ``scipy.signal.peak_widths`` at ``rel_height=0.5`` which
    measures the width of each peak at 50 % of its prominence.  This is
    a robust signal-processing metric — it is NOT equivalent to clinical
    QRS delineation (which requires wavelet or template-matching
    methods) but provides a physiologically plausible proxy for the
    dominant deflection width.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal (1-D).
    peak_indices : np.ndarray
        Detected R-peak sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Approximate peak width per beat in milliseconds.
    """
    if len(peak_indices) == 0:
        return np.array([], dtype=np.float64)

    # peak_widths returns (widths_in_samples, width_heights, left_ips, right_ips)
    widths_samples, _, _, _ = peak_widths(signal, peak_indices, rel_height=0.5)

    # Convert from samples to milliseconds
    widths_ms = widths_samples * 1000.0 / fs

    return widths_ms


# ── Aggregate entry point ──────────────────────────────────────────────


def extract_features(signal: np.ndarray, peak_indices: np.ndarray,
                     fs: float) -> ECGFeatures:
    """Compute all clinical features for an ECG record.

    This is the single entry point called by main.py.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal (1-D).
    peak_indices : np.ndarray
        Detected R-peak sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    ECGFeatures
        Populated feature dataclass.
    """
    rr = compute_rr_intervals(peak_indices, fs)
    ihr = compute_instantaneous_hr(rr)

    mean_hr = float(np.nanmean(ihr)) if len(ihr) > 0 else 0.0
    median_hr = float(np.nanmedian(ihr)) if len(ihr) > 0 else 0.0

    sdnn = compute_sdnn(rr)
    rmssd = compute_rmssd(rr)
    pnn50 = compute_pnn50(rr)

    qrs_dur = estimate_peak_width(signal, peak_indices, fs)

    pw_std = float(np.std(qrs_dur)) if len(qrs_dur) > 1 else 0.0

    return ECGFeatures(
        rr_intervals=rr,
        instantaneous_hr=ihr,
        mean_hr=mean_hr,
        median_hr=median_hr,
        sdnn=sdnn,
        rmssd=rmssd,
        pnn50=pnn50,
        qrs_duration_ms=qrs_dur,
        peak_width_std_ms=pw_std,
    )


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate peaks at ~72 BPM (0.833 s intervals) with slight jitter.
    _fs = 360.0
    _base_interval = int(0.833 * _fs)  # ~300 samples
    _peaks = np.array([_base_interval * i for i in range(1, 20)])
    _peaks = _peaks + np.random.randint(-5, 6, size=len(_peaks))

    # Dummy signal (flat + spikes at peaks).
    _sig = np.zeros(int(20 * _fs))
    for _p in _peaks:
        if _p < len(_sig):
            _sig[_p] = 3.0

    _feats = extract_features(_sig, _peaks, _fs)

    print(f"RR intervals : {len(_feats.rr_intervals)}")
    print(f"Mean HR      : {_feats.mean_hr:.1f} BPM")
    print(f"Median HR    : {_feats.median_hr:.1f} BPM")
    print(f"SDNN         : {_feats.sdnn * 1000:.1f} ms")
    print(f"RMSSD        : {_feats.rmssd * 1000:.1f} ms")
    print(f"pNN50        : {_feats.pnn50:.1f} %")
    print(f"QRS durations: mean {np.mean(_feats.qrs_duration_ms):.1f} ms, "
          f"std {_feats.peak_width_std_ms:.1f} ms")
    print("features.py OK")
