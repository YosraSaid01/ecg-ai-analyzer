"""
preprocess.py — ECG Signal Preprocessing

Standard biomedical signal processing pipeline for ECG waveforms:
    1. Baseline wander removal  (high-pass filter at ~0.5 Hz)
    2. Bandpass filtering        (0.5–40 Hz — standard clinical ECG range)
    3. Amplitude normalization   (zero-mean, unit-variance)

All functions accept and return 1-D NumPy arrays so they can be
composed freely or used independently.

References:
    - Sörnmo L, Laguna P. Bioelectrical Signal Processing in Cardiac
      and Neurological Applications. Academic Press, 2005.
    - Pan J, Tompkins WJ. A Real-Time QRS Detection Algorithm.
      IEEE Trans Biomed Eng. 1985;32(3):230-236.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ── Filter design helpers ──────────────────────────────────────────────


def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (applied twice via filtfilt → effective order is 2×).

    Returns
    -------
    b, a : ndarray
        Numerator and denominator coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def _butter_highpass(cutoff: float, fs: float, order: int = 4):
    """Design a Butterworth high-pass filter.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    b, a : ndarray
        Filter coefficients.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return b, a


# ── Public preprocessing functions ─────────────────────────────────────


def remove_baseline_wander(signal: np.ndarray, fs: float,
                           cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    """Remove baseline wander using a zero-phase high-pass filter.

    Baseline wander is a low-frequency artifact (< 0.5 Hz) caused by
    respiration and electrode motion.  A high-pass Butterworth filter
    applied with ``filtfilt`` (zero-phase) removes the drift without
    distorting QRS morphology.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal (1-D).
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        High-pass cutoff frequency in Hz (default 0.5).
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Signal with baseline wander removed.
    """
    b, a = _butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, signal).astype(np.float64)


def bandpass_filter(signal: np.ndarray, fs: float,
                    lowcut: float = 0.5, highcut: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    The 0.5–40 Hz band retains the clinically relevant ECG components
    (P-wave, QRS complex, T-wave) while suppressing high-frequency EMG
    noise and low-frequency baseline drift.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal (1-D).
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff in Hz (default 0.5).
    highcut : float
        Upper cutoff in Hz (default 40.0).
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal.
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal).astype(np.float64)


def remove_powerline_noise(signal: np.ndarray, fs: float,
                           freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Remove power-line interference with a notch filter.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal (1-D).
    fs : float
        Sampling frequency in Hz.
    freq : float
        Power-line frequency (50 Hz in Europe, 60 Hz in North America).
    quality : float
        Quality factor of the notch filter.

    Returns
    -------
    np.ndarray
        Signal with power-line component attenuated.
    """
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal).astype(np.float64)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize the signal (zero mean, unit variance).

    Normalization ensures amplitude-independent peak detection and
    makes thresholds transferable across records with different gains.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1-D).

    Returns
    -------
    np.ndarray
        Normalized signal.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-10:
        # Avoid division by zero for flat signals.
        return signal - mean
    return (signal - mean) / std


def preprocess_ecg(signal: np.ndarray, fs: float,
                   lowcut: float = 0.5, highcut: float = 40.0,
                   notch_freq: float = None) -> np.ndarray:
    """Full preprocessing pipeline: filter → (optional notch) → normalize.

    This is the recommended single-call entry point used by main.py.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal (1-D, physical units from WFDB).
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Bandpass lower cutoff (default 0.5 Hz).
    highcut : float
        Bandpass upper cutoff (default 40.0 Hz).
    notch_freq : float or None
        If provided, apply a notch filter at this frequency (e.g. 50 or 60 Hz).

    Returns
    -------
    np.ndarray
        Preprocessed, normalized ECG signal.
    """
    filtered = bandpass_filter(signal, fs, lowcut=lowcut, highcut=highcut)

    if notch_freq is not None:
        filtered = remove_powerline_noise(filtered, fs, freq=notch_freq)

    return normalize(filtered)


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with a synthetic signal.
    _fs = 360.0
    _t = np.arange(0, 5, 1 / _fs)
    # Simulate: 1 Hz baseline drift + 10 Hz "QRS-like" component + noise
    _raw = 0.5 * np.sin(2 * np.pi * 0.3 * _t) + np.sin(2 * np.pi * 10 * _t) + 0.2 * np.random.randn(len(_t))

    _clean = preprocess_ecg(_raw, _fs)
    print(f"Input  — mean: {np.mean(_raw):.4f}, std: {np.std(_raw):.4f}")
    print(f"Output — mean: {np.mean(_clean):.4f}, std: {np.std(_clean):.4f}")
    print("preprocess.py OK")
