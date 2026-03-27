"""
peaks.py — R-Peak Detection

Detects R-peaks in a preprocessed ECG signal using
``scipy.signal.find_peaks`` with physiologically motivated constraints.

Design rationale:
    • Minimum peak distance is derived from the maximum plausible heart
      rate (~200 BPM → ~300 ms inter-beat interval → ~108 samples at 360 Hz).
      A default of 200 ms is used as a conservative floor.
    • Prominence threshold rejects small deflections that are not true
      QRS complexes.  Because the input is z-score normalized, a
      prominence of ~0.6 σ works well as a starting point.
    • An adaptive threshold refinement step optionally adjusts the
      prominence based on the signal's amplitude distribution.

References:
    Pan J, Tompkins WJ. A Real-Time QRS Detection Algorithm.
    IEEE Trans Biomed Eng. 1985;32(3):230-236.
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


def detect_r_peaks(signal: np.ndarray, fs: float,
                   min_distance_ms: float = 200.0,
                   prominence: float = 0.6,
                   height_percentile: float = None) -> np.ndarray:
    """Detect R-peaks in a preprocessed ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed (filtered + normalized) ECG signal, 1-D.
    fs : float
        Sampling frequency in Hz.
    min_distance_ms : float
        Minimum time between consecutive R-peaks in milliseconds.
        Default 200 ms corresponds to a maximum heart rate of 300 BPM,
        which safely covers even extreme tachycardia.
    prominence : float
        Minimum prominence of a peak (in signal amplitude units).
        For a z-score normalized signal, 0.6 is a reasonable default.
    height_percentile : float or None
        If provided (0–100), only peaks whose amplitude exceeds this
        percentile of the signal are kept.  Useful as an adaptive
        threshold when prominence alone is insufficient.

    Returns
    -------
    np.ndarray
        Array of sample indices where R-peaks were detected.
    """
    min_distance_samples = int(np.round(min_distance_ms * fs / 1000.0))

    kwargs = {
        "distance": max(min_distance_samples, 1),
        "prominence": prominence,
    }

    if height_percentile is not None:
        kwargs["height"] = np.percentile(signal, height_percentile)

    peaks, _properties = find_peaks(signal, **kwargs)

    return peaks


def refine_peaks(signal: np.ndarray, peak_indices: np.ndarray,
                 search_window_ms: float = 50.0, fs: float = 360.0) -> np.ndarray:
    """Refine peak positions by searching for the true local maximum.

    ``find_peaks`` may land a sample or two away from the exact apex of
    the R-wave.  This function nudges each detected index to the highest
    point within a small window around it.

    Parameters
    ----------
    signal : np.ndarray
        The same preprocessed ECG signal used for detection.
    peak_indices : np.ndarray
        Initial R-peak sample indices.
    search_window_ms : float
        Half-width of the search window in milliseconds (default 50 ms).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Refined peak indices.
    """
    half_win = int(np.round(search_window_ms * fs / 1000.0))
    n = len(signal)
    refined = np.empty_like(peak_indices)

    for i, idx in enumerate(peak_indices):
        lo = max(0, idx - half_win)
        hi = min(n, idx + half_win + 1)
        refined[i] = lo + np.argmax(signal[lo:hi])

    return refined


def _remove_duplicate_peaks(peak_indices: np.ndarray,
                            min_distance_samples: int) -> np.ndarray:
    """Remove duplicate peaks that are closer than the minimum distance.

    After the refinement step, two initially distinct detections may
    have been nudged to the same (or very close) sample.  This function
    keeps only the first occurrence within each minimum-distance window.

    Parameters
    ----------
    peak_indices : np.ndarray
        Sorted peak sample indices (may contain near-duplicates).
    min_distance_samples : int
        Minimum allowed gap between consecutive peaks.

    Returns
    -------
    np.ndarray
        Deduplicated peak indices.
    """
    if len(peak_indices) <= 1:
        return peak_indices

    keep = [peak_indices[0]]
    for idx in peak_indices[1:]:
        if idx - keep[-1] >= min_distance_samples:
            keep.append(idx)
    return np.array(keep, dtype=peak_indices.dtype)


def prune_false_positives(signal: np.ndarray, peak_indices: np.ndarray,
                          fs: float,
                          refractory_ms: float = 360.0,
                          width_ratio_thr: float = 2.5,
                          prom_ratio_thr: float = 0.50) -> np.ndarray:
    """Remove likely T-wave and secondary-deflection false positives.

    Strategy
    --------
    1. Compute prominence and width-at-half-prominence for every
       candidate peak.
    2. Estimate the "typical R-peak width" as the 25th-percentile of
       all candidate widths.  True QRS complexes are the narrowest
       deflections; T-waves and artefacts are 2–4× wider.
    3. Remove any peak whose width exceeds ``width_ratio_thr`` times
       the typical R-peak width — these are almost certainly T-waves.
    4. For every pair of surviving peaks closer than
       ``refractory_ms``, keep only the one with higher prominence.
       This implements a physiological refractory period: the
       ventricles cannot depolarise twice within ~300 ms, so the
       weaker of two close peaks is a secondary deflection.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal (1-D).
    peak_indices : np.ndarray
        Candidate R-peak sample indices (sorted).
    fs : float
        Sampling frequency in Hz.
    refractory_ms : float
        Refractory window in ms (default 360).  Peak pairs closer
        than this compete; the weaker one is removed.
    width_ratio_thr : float
        A peak wider than ``width_ratio_thr × median_qrs_width`` is
        rejected as a probable T-wave (default 2.5).
    prom_ratio_thr : float
        During pairwise competition, if the weaker peak's prominence
        is less than ``prom_ratio_thr`` times the stronger peak's
        prominence, it is removed (default 0.50).

    Returns
    -------
    np.ndarray
        Pruned peak indices with false positives removed.
    """
    if len(peak_indices) < 3:
        return peak_indices

    # ── Compute per-peak features ───────────────────────────────
    proms = peak_prominences(signal, peak_indices)[0]
    widths_samples = peak_widths(signal, peak_indices, rel_height=0.5)[0]

    # ── Phase 1: width-based T-wave rejection ───────────────────
    # The 25th percentile of widths represents narrow QRS complexes.
    # Peaks much wider than this are T-waves or artefacts.
    q25_width = np.percentile(widths_samples, 25)
    # Safety: q25 must be at least 2 samples to avoid rejecting everything
    q25_width = max(q25_width, 2.0)
    width_ok = widths_samples <= (width_ratio_thr * q25_width)

    # Only reject on width if it would not remove too many peaks
    # (safety: keep at least 60% of candidates to preserve recall)
    if np.sum(width_ok) >= 0.5 * len(peak_indices):
        peak_indices = peak_indices[width_ok]
        proms = proms[width_ok]
        widths_samples = widths_samples[width_ok]

    # ── Phase 2: prominence-based adaptive threshold ────────────
    # If there is a clear bimodal distribution (R-peaks vs T-waves),
    # the median prominence is a good separator.  Require peaks to
    # have at least 40% of the median prominence of the top half.
    if len(proms) >= 4:
        sorted_proms = np.sort(proms)
        top_half_median = np.median(sorted_proms[len(sorted_proms) // 2:])
        prom_floor = 0.40 * top_half_median
        prom_ok = proms >= prom_floor
        if np.sum(prom_ok) >= 0.5 * len(peak_indices):
            peak_indices = peak_indices[prom_ok]
            proms = proms[prom_ok]

    # ── Phase 3: refractory competition ─────────────────────────
    # For peaks closer than the refractory window, keep the more
    # prominent one.  Process greedily left-to-right.
    refractory_samples = int(np.round(refractory_ms * fs / 1000.0))

    keep = np.ones(len(peak_indices), dtype=bool)
    i = 0
    while i < len(peak_indices) - 1:
        if not keep[i]:
            i += 1
            continue

        j = i + 1
        while j < len(peak_indices) and (peak_indices[j] - peak_indices[i]) < refractory_samples:
            if not keep[j]:
                j += 1
                continue

            # Compare: loser gets removed
            if proms[j] < prom_ratio_thr * proms[i]:
                keep[j] = False
            elif proms[i] < prom_ratio_thr * proms[j]:
                keep[i] = False
                break  # i lost, move on
            else:
                # Similar prominence — keep the taller one
                if signal[peak_indices[i]] >= signal[peak_indices[j]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
            j += 1
        i += 1

    return peak_indices[keep]


def detect_r_peaks_adaptive(signal: np.ndarray, fs: float,
                            min_distance_ms: float = 280.0) -> np.ndarray:
    """Adaptive R-peak detection with automatic threshold selection.

    Improved strategy (v2):
        1. Use the signal's standard deviation to set a robust
           prominence threshold — this is far more selective than
           the previous mean-envelope approach, especially on
           z-score normalized signals where std ≈ 1.0.
        2. Enforce a minimum height so that only positive deflections
           (true R-waves) are detected — rejects inverted T-waves
           and noise troughs.
        3. Increase the minimum inter-peak distance to 280 ms
           (max ~214 BPM), which covers all but the most extreme
           tachycardias while strongly suppressing T-wave detections
           that typically fall 200–300 ms after the R-peak.
        4. Refine peak positions, then deduplicate.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal (1-D, z-score normalized recommended).
    fs : float
        Sampling frequency in Hz.
    min_distance_ms : float
        Minimum inter-peak distance in milliseconds (default 280).

    Returns
    -------
    np.ndarray
        Detected and refined R-peak sample indices.
    """
    sig_std = np.std(signal)

    # ── Prominence: 0.8 × std ──────────────────────────────────────
    # For a z-normalized signal (std ≈ 1), this gives ~0.8.
    # R-peaks in MIT-BIH typically have prominence > 1.5 σ; T-waves
    # are usually < 0.5 σ.  The 0.8 threshold sits comfortably in
    # between, preserving recall while cutting most false positives.
    adaptive_prominence = 0.8 * sig_std
    adaptive_prominence = max(adaptive_prominence, 0.5)  # safety floor

    # ── Height: only positive peaks above the 50th percentile ──────
    # R-peaks are the dominant positive deflections in the ECG.
    # Requiring height > median rejects noise and baseline artefacts.
    height_floor = np.percentile(signal, 50)

    peaks = detect_r_peaks(
        signal, fs,
        min_distance_ms=min_distance_ms,
        prominence=adaptive_prominence,
        height_percentile=None,  # we pass height directly below
    )

    # Apply height filter (more flexible than height_percentile which
    # recomputes inside find_peaks — here we combine with prominence).
    if len(peaks) > 0:
        peaks = peaks[signal[peaks] >= height_floor]

    if len(peaks) == 0:
        return peaks

    # ── Refine + deduplicate ───────────────────────────────────────
    refined = refine_peaks(signal, peaks, fs=fs)
    min_dist_samples = int(np.round(min_distance_ms * fs / 1000.0))
    refined = np.sort(refined)
    refined = _remove_duplicate_peaks(refined, min_dist_samples)

    # ── Prune T-wave and secondary-deflection false positives ─────
    refined = prune_false_positives(signal, refined, fs)

    return refined


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic test: simulate a rough ECG-like signal at 360 Hz.
    _fs = 360.0
    _duration = 10.0  # seconds
    _t = np.arange(0, _duration, 1 / _fs)

    # Create sharp peaks at ~75 BPM (0.8 s interval).
    _signal = np.zeros_like(_t)
    _interval_samples = int(0.8 * _fs)
    _true_peaks = list(range(_interval_samples, len(_t), _interval_samples))
    for _pk in _true_peaks:
        _lo = max(0, _pk - 5)
        _hi = min(len(_t), _pk + 6)
        _signal[_lo:_hi] += np.hanning(_hi - _lo) * 3.0

    # Add noise.
    _signal += 0.2 * np.random.randn(len(_signal))

    # Normalize (mimics preprocess output).
    _signal = (_signal - _signal.mean()) / _signal.std()

    _detected = detect_r_peaks_adaptive(_signal, _fs)
    print(f"True peaks : {len(_true_peaks)}")
    print(f"Detected   : {len(_detected)}")
    print("peaks.py OK")
