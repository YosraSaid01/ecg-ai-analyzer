"""
data_loader.py — MIT-BIH Arrhythmia Database Loader

Loads ECG signals and expert annotations from local MIT-BIH records
using the WFDB Python library.

Expected file layout:
    data/mitdb/<record_id>.dat
    data/mitdb/<record_id>.hea
    data/mitdb/<record_id>.atr

Reference:
    Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
    IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
"""

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import wfdb


# Standard MIT-BIH beat-type annotation symbols (subset).
# Full list: https://www.physionet.org/physiobank/annotations.shtml
BEAT_ANNOTATION_SYMBOLS = {
    "N",  # Normal beat
    "L",  # Left bundle branch block beat
    "R",  # Right bundle branch block beat
    "A",  # Atrial premature beat
    "V",  # Premature ventricular contraction
    "/",  # Paced beat
    "F",  # Fusion of ventricular and normal beat
    "f",  # Fusion of paced and normal beat
    "j",  # Nodal (junctional) escape beat
    "a",  # Aberrated atrial premature beat
    "E",  # Ventricular escape beat
    "J",  # Nodal (junctional) premature beat
    "S",  # Supraventricular premature beat
    "e",  # Atrial escape beat
    "Q",  # Unclassifiable beat
}


@dataclass
class ECGRecord:
    """Container for a single MIT-BIH ECG record and its annotations.

    Attributes:
        record_id: MIT-BIH record identifier (e.g. "100").
        signal: 1-D NumPy array of ECG samples (MLII lead).
        fs: Sampling frequency in Hz.
        ann_indices: Sample indices of expert-annotated beats.
        ann_symbols: Annotation symbol for each beat (same length as ann_indices).
        units: Physical unit string for the signal channel.
        duration_sec: Total signal duration in seconds.
    """
    record_id: str
    signal: np.ndarray
    fs: float
    ann_indices: np.ndarray
    ann_symbols: List[str]
    units: str = ""
    duration_sec: float = 0.0


def resolve_data_path(record_id: str, data_dir: str = None) -> str:
    """Resolve the full path prefix for a MIT-BIH record.

    Parameters
    ----------
    record_id : str
        Record identifier, e.g. "100".
    data_dir : str, optional
        Directory containing the mitdb files.  Defaults to
        ``<project_root>/data/mitdb``.

    Returns
    -------
    str
        Full path prefix suitable for wfdb.rdrecord / wfdb.rdann.

    Raises
    ------
    FileNotFoundError
        If the expected .hea file does not exist at the resolved path.
    """
    if data_dir is None:
        # Assume standard project layout: app/ is one level below project root.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data", "mitdb")

    record_path = os.path.join(data_dir, record_id)

    # Quick sanity check — wfdb needs at least the .hea file.
    hea_file = record_path + ".hea"
    if not os.path.isfile(hea_file):
        raise FileNotFoundError(
            f"MIT-BIH header file not found: {hea_file}\n"
            f"Download the database with:\n"
            f"  python -c \"import wfdb; wfdb.dl_database('mitdb', '{data_dir}')\""
        )

    return record_path


def load_record(record_id: str, channel: int = 0, data_dir: str = None) -> ECGRecord:
    """Load a MIT-BIH record and its reference annotations.

    Parameters
    ----------
    record_id : str
        Record identifier (e.g. "100", "201").
    channel : int, optional
        Signal channel index.  Default ``0`` corresponds to the MLII lead
        for most MIT-BIH records.
    data_dir : str, optional
        Path to the directory containing the WFDB files.

    Returns
    -------
    ECGRecord
        Populated dataclass with signal, annotations, and metadata.
    """
    record_path = resolve_data_path(record_id, data_dir)

    # --- Load signal --------------------------------------------------------
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, channel].astype(np.float64)
    fs = float(record.fs)
    units = record.units[channel] if record.units else "mV"

    # --- Load annotations ---------------------------------------------------
    annotation = wfdb.rdann(record_path, "atr")
    ann_indices = np.array(annotation.sample, dtype=np.int64)
    ann_symbols = list(annotation.symbol)

    # --- Build output -------------------------------------------------------
    duration_sec = len(signal) / fs

    return ECGRecord(
        record_id=record_id,
        signal=signal,
        fs=fs,
        ann_indices=ann_indices,
        ann_symbols=ann_symbols,
        units=units,
        duration_sec=duration_sec,
    )


def filter_beat_annotations(ecg: ECGRecord) -> ECGRecord:
    """Return a copy of the record keeping only true beat annotations.

    Non-beat symbols (e.g. rhythm change markers '+', '~', '|') are
    removed so that peak-detection evaluation compares only beats.

    Parameters
    ----------
    ecg : ECGRecord
        Original record with all annotation types.

    Returns
    -------
    ECGRecord
        New record with non-beat annotations filtered out.
    """
    mask = [sym in BEAT_ANNOTATION_SYMBOLS for sym in ecg.ann_symbols]
    beat_indices = ecg.ann_indices[mask]
    beat_symbols = [s for s, keep in zip(ecg.ann_symbols, mask) if keep]

    return ECGRecord(
        record_id=ecg.record_id,
        signal=ecg.signal,
        fs=ecg.fs,
        ann_indices=beat_indices,
        ann_symbols=beat_symbols,
        units=ecg.units,
        duration_sec=ecg.duration_sec,
    )


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _rec = load_record("100")
    _rec = filter_beat_annotations(_rec)
    print(f"Record : {_rec.record_id}")
    print(f"Samples: {len(_rec.signal):,}  |  Fs: {_rec.fs} Hz  |  Duration: {_rec.duration_sec:.1f} s")
    print(f"Beats  : {len(_rec.ann_indices):,}")
    print(f"Symbols: {set(_rec.ann_symbols)}")
