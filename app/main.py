"""
main.py — ECG AI Analyzer Pipeline

Full end-to-end pipeline:
    1. Load MIT-BIH record and annotations
    2. Preprocess the ECG signal
    3. Detect R-peaks
    4. Compute clinical features
    5. Apply rule-based abnormality detection
    6. Evaluate detection against expert annotations
    7. Visualize results
    8. Generate and print clinical explanation

Usage:
    python app/main.py
    python app/main.py --record 101 --start 5 --duration 15

Requires MIT-BIH data files in:  data/mitdb/
Download with:
    python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
"""

import argparse
import os
import sys

# Ensure the app/ directory is on the import path when running as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_record, filter_beat_annotations
from preprocess import preprocess_ecg
from peaks import detect_r_peaks_adaptive
from features import extract_features
from rules import apply_rules
from evaluate import evaluate_detection, print_eval_report
from visualize import show_all
from llm_explainer import generate_explanation


def run_pipeline(record_id: str = "100",
                 start_sec: float = 0.0,
                 duration_sec: float = 10.0,
                 save_plots: bool = False) -> None:
    """Execute the complete ECG analysis pipeline.

    Parameters
    ----------
    record_id : str
        MIT-BIH record identifier (e.g. "100", "101", "201").
    start_sec : float
        Start of the visualization window in seconds.
    duration_sec : float
        Duration of the visualization window in seconds.
    save_plots : bool
        If True, save plot PNGs to the project ``outputs/`` directory.
    """
    print(f"\n{'=' * 60}")
    print(f"  ECG AI Analyzer — Record {record_id}")
    print(f"{'=' * 60}\n")

    # ── 1. Load record ─────────────────────────────────────────────
    print("[1/8] Loading MIT-BIH record …")
    ecg = load_record(record_id)
    ecg = filter_beat_annotations(ecg)
    print(f"      Samples: {len(ecg.signal):,}  |  Fs: {ecg.fs} Hz  |  "
          f"Duration: {ecg.duration_sec:.1f} s")
    print(f"      Expert beat annotations: {len(ecg.ann_indices):,}")

    raw_signal = ecg.signal.copy()

    # ── 2. Preprocess ──────────────────────────────────────────────
    print("[2/8] Preprocessing (bandpass 0.5–40 Hz + normalization) …")
    processed = preprocess_ecg(ecg.signal, ecg.fs)
    print("      Done.")

    # ── 3. Detect R-peaks ──────────────────────────────────────────
    print("[3/8] Detecting R-peaks (adaptive threshold) …")
    detected_peaks = detect_r_peaks_adaptive(processed, ecg.fs)
    print(f"      Detected: {len(detected_peaks):,} peaks")

    # ── 4. Compute features ────────────────────────────────────────
    print("[4/8] Computing clinical features …")
    feats = extract_features(processed, detected_peaks, ecg.fs)
    print(f"      Mean HR    : {feats.mean_hr:.1f} BPM")
    print(f"      SDNN       : {feats.sdnn * 1000:.1f} ms")
    print(f"      RMSSD      : {feats.rmssd * 1000:.1f} ms")
    print(f"      pNN50      : {feats.pnn50:.1f} %")
    if feats.qrs_duration_ms is not None and len(feats.qrs_duration_ms) > 0:
        import numpy as _np
        print(f"      Peak Width : {_np.mean(feats.qrs_duration_ms):.1f} "
              f"± {feats.peak_width_std_ms:.1f} ms  "
              f"(half-prominence proxy)")

    # ── 5. Apply rules ─────────────────────────────────────────────
    print("[5/8] Applying rule-based abnormality detection …")
    rule_results = apply_rules(feats)
    if rule_results.is_normal:
        print("      No abnormalities flagged.")
    else:
        for flag in rule_results.flags:
            print(f"      [{flag.severity.upper()}] {flag.name}: "
                  f"{flag.description}")

    # ── 6. Evaluate against annotations ────────────────────────────
    print("[6/8] Evaluating detection against expert annotations …")
    eval_metrics = evaluate_detection(
        detected_peaks, ecg.ann_indices, ecg.fs, tolerance_ms=150.0
    )
    report = print_eval_report(eval_metrics, record_id=record_id)
    print(report)

    # ── 7. Visualize ───────────────────────────────────────────────
    print("[7/8] Generating plots …")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "outputs")
    os.makedirs(save_dir, exist_ok=True)
    if save_plots:
        print(f"      Saving to: {save_dir}")

    show_all(
        signal_raw=raw_signal,
        signal_processed=processed,
        fs=ecg.fs,
        detected_peaks=detected_peaks,
        annotation_peaks=ecg.ann_indices,
        feats=feats,
        rule_results=rule_results,
        record_id=record_id,
        start_sec=start_sec,
        duration_sec=duration_sec,
        save_dir=save_dir,
        eval_metrics=eval_metrics,
    )

    # ── 8. Generate explanation ────────────────────────────────────
    print("[8/8] Generating clinical explanation …\n")
    explanation = generate_explanation(
        feats, rule_results, eval_metrics, record_id=record_id
    )
    print(explanation)

    print("Pipeline complete.\n")


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="ECG AI Analyzer — MIT-BIH Signal Processing Pipeline"
    )
    parser.add_argument(
        "--record", type=str, default="100",
        help="MIT-BIH record ID (default: 100)"
    )
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="Visualization start time in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Visualization window duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save plot images to outputs/ directory"
    )

    args = parser.parse_args()

    run_pipeline(
        record_id=args.record,
        start_sec=args.start,
        duration_sec=args.duration,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
