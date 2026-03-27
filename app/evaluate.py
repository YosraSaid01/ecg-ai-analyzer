"""
evaluate.py — Peak Detection Performance Evaluation

Compares detected R-peaks against expert annotations from the MIT-BIH
database and computes standard detection metrics.

Matching strategy:
    For each annotation, find the closest detected peak within a
    tolerance window.  If a match is found, both are consumed (greedy
    one-to-one matching to prevent double counting).

Metrics:
    • True Positives  (TP) — annotation matched by a detected peak
    • False Positives (FP) — detected peak with no matching annotation
    • False Negatives (FN) — annotation with no matching detected peak
    • Precision = TP / (TP + FP)
    • Recall    = TP / (TP + FN)
    • F1-score  = 2 · Precision · Recall / (Precision + Recall)

References:
    ANSI/AAMI EC57:2012 — Testing and reporting performance results of
    cardiac rhythm and ST segment measurement algorithms.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EvalMetrics:
    """Container for peak-detection evaluation results.

    Attributes
    ----------
    tp : int
        True positives — correctly detected beats.
    fp : int
        False positives — spurious detections.
    fn : int
        False negatives — missed beats.
    precision : float
        TP / (TP + FP).  1.0 when there are no false alarms.
    recall : float
        TP / (TP + FN).  1.0 when every beat is detected.
    f1 : float
        Harmonic mean of precision and recall.
    tolerance_ms : float
        Tolerance window used for matching (in milliseconds).
    mean_offset_ms : float
        Mean signed offset (detected − annotation) for matched pairs,
        in milliseconds.  Positive = detection is late.
    std_offset_ms : float
        Standard deviation of the offset, in milliseconds.
    """
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    tolerance_ms: float
    mean_offset_ms: float = 0.0
    std_offset_ms: float = 0.0


def match_peaks(detected: np.ndarray, reference: np.ndarray,
                tolerance_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one greedy matching of detected peaks to reference annotations.

    Algorithm:
        1. For each reference beat (in order), find the closest detected
           peak within ± ``tolerance_samples``.
        2. If a match exists, record it and remove the detected peak from
           the candidate pool so it cannot be matched again.
        3. Unmatched reference beats are false negatives; unmatched
           detected peaks are false positives.

    Parameters
    ----------
    detected : np.ndarray
        Sorted sample indices of detected R-peaks.
    reference : np.ndarray
        Sorted sample indices of expert-annotated beats.
    tolerance_samples : int
        Maximum allowed distance (in samples) between a detected peak
        and an annotation for them to be considered a match.

    Returns
    -------
    tp_pairs : np.ndarray, shape (N, 2)
        Matched pairs as ``[detected_index, reference_index]``.
    fp_indices : np.ndarray
        Sample indices of unmatched detected peaks.
    fn_indices : np.ndarray
        Sample indices of unmatched reference annotations.
    """
    detected = np.sort(detected)
    reference = np.sort(reference)

    matched_det = set()
    matched_ref = set()
    tp_pairs = []

    for r_idx, r_pos in enumerate(reference):
        # Binary search for the closest detected peak.
        insert_pos = np.searchsorted(detected, r_pos)

        best_dist = tolerance_samples + 1
        best_d_idx = -1

        # Check the candidate at insert_pos and insert_pos - 1.
        for candidate in (insert_pos - 1, insert_pos):
            if candidate < 0 or candidate >= len(detected):
                continue
            if candidate in matched_det:
                continue
            dist = abs(int(detected[candidate]) - int(r_pos))
            if dist <= tolerance_samples and dist < best_dist:
                best_dist = dist
                best_d_idx = candidate

        if best_d_idx >= 0:
            tp_pairs.append((detected[best_d_idx], r_pos))
            matched_det.add(best_d_idx)
            matched_ref.add(r_idx)

    tp_pairs = np.array(tp_pairs, dtype=np.int64).reshape(-1, 2) if tp_pairs else np.empty((0, 2), dtype=np.int64)

    fp_mask = np.array([i not in matched_det for i in range(len(detected))])
    fn_mask = np.array([i not in matched_ref for i in range(len(reference))])

    fp_indices = detected[fp_mask]
    fn_indices = reference[fn_mask]

    return tp_pairs, fp_indices, fn_indices


def evaluate_detection(detected: np.ndarray, reference: np.ndarray,
                       fs: float, tolerance_ms: float = 150.0) -> EvalMetrics:
    """Evaluate R-peak detection performance against expert annotations.

    Parameters
    ----------
    detected : np.ndarray
        Detected R-peak sample indices.
    reference : np.ndarray
        Expert-annotated beat sample indices (ground truth).
    fs : float
        Sampling frequency in Hz.
    tolerance_ms : float
        Maximum allowed offset between a detected peak and an
        annotation, in milliseconds.  AAMI EC57 recommends 150 ms.

    Returns
    -------
    EvalMetrics
        Precision, recall, F1, and offset statistics.
    """
    tolerance_samples = int(np.round(tolerance_ms * fs / 1000.0))

    tp_pairs, fp_indices, fn_indices = match_peaks(detected, reference,
                                                   tolerance_samples)

    tp = len(tp_pairs)
    fp = len(fp_indices)
    fn = len(fn_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    # Offset statistics (detected − reference) for matched pairs.
    if tp > 0:
        offsets_samples = tp_pairs[:, 0].astype(np.float64) - tp_pairs[:, 1].astype(np.float64)
        offsets_ms = offsets_samples * 1000.0 / fs
        mean_offset = float(np.mean(offsets_ms))
        std_offset = float(np.std(offsets_ms))
    else:
        mean_offset = 0.0
        std_offset = 0.0

    return EvalMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        tolerance_ms=tolerance_ms,
        mean_offset_ms=mean_offset,
        std_offset_ms=std_offset,
    )


def print_eval_report(metrics: EvalMetrics, record_id: str = "") -> str:
    """Format evaluation metrics as a human-readable report string.

    Parameters
    ----------
    metrics : EvalMetrics
        Computed evaluation metrics.
    record_id : str, optional
        Record identifier for the report header.

    Returns
    -------
    str
        Formatted multi-line report.
    """
    header = f"Record {record_id}" if record_id else "Evaluation"
    lines = [
        f"{'=' * 50}",
        f"  R-Peak Detection Evaluation — {header}",
        f"{'=' * 50}",
        f"  Tolerance      : ±{metrics.tolerance_ms:.0f} ms",
        f"  True Positives : {metrics.tp:>6d}",
        f"  False Positives: {metrics.fp:>6d}",
        f"  False Negatives: {metrics.fn:>6d}",
        f"  {'─' * 46}",
        f"  Precision      : {metrics.precision:>9.4f}",
        f"  Recall         : {metrics.recall:>9.4f}",
        f"  F1-score       : {metrics.f1:>9.4f}",
        f"  {'─' * 46}",
        f"  Mean offset    : {metrics.mean_offset_ms:>+7.1f} ms",
        f"  Offset std     : {metrics.std_offset_ms:>7.1f} ms",
        f"{'=' * 50}",
    ]
    return "\n".join(lines)


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    _fs = 360.0

    # Simulated ground truth at regular 0.8 s intervals.
    _reference = np.arange(200, 5000, int(0.8 * _fs))

    # Simulated detections: most correct, a few missed, a few extra.
    _detected = _reference.copy().astype(np.int64)
    # Add small jitter.
    _detected = _detected + np.random.randint(-10, 11, size=len(_detected))
    # Remove two beats (will become FN).
    _detected = np.delete(_detected, [3, 7])
    # Add a spurious peak (will become FP).
    _detected = np.append(_detected, [2500])
    _detected = np.sort(_detected)

    _metrics = evaluate_detection(_detected, _reference, _fs, tolerance_ms=150.0)
    print(print_eval_report(_metrics, record_id="synthetic"))
    print("evaluate.py OK")
