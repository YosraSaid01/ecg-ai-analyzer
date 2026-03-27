"""
llm_explainer.py — Rule-Based Clinical Narrative Generator

Generates a structured, clinician-style textual explanation based
ENTIRELY on computed features and rule-engine flags.

This module does NOT call any external API or language model.
It uses template-based generation with conditional logic to produce
a readable summary that a clinician could review alongside the ECG.

IMPORTANT — Scope of output:
    • Informational only — never diagnostic.
    • Uses hedging language ("may be consistent with", "suggests",
      "warrants further review").
    • Always ends with a disclaimer.
"""

from typing import List

import numpy as np

from features import ECGFeatures
from rules import RuleResults, AbnormalityFlag
from evaluate import EvalMetrics


# ── Section generators ─────────────────────────────────────────────────


def _section_header(title: str) -> str:
    """Produce a formatted section header."""
    return f"\n{'─' * 50}\n  {title}\n{'─' * 50}"


def _rate_summary(feats: ECGFeatures) -> str:
    """Generate a narrative paragraph about heart rate."""
    lines = []
    lines.append(f"The mean heart rate across the analyzed segment is "
                 f"{feats.mean_hr:.0f} BPM "
                 f"(median {feats.median_hr:.0f} BPM).")

    if feats.mean_hr > 100:
        lines.append("This is above the standard threshold of 100 BPM "
                      "and may be consistent with sinus tachycardia or "
                      "another tachyarrhythmia.")
    elif feats.mean_hr < 60:
        lines.append("This is below the standard threshold of 60 BPM "
                      "and may be consistent with sinus bradycardia.  "
                      "In trained athletes, a resting rate below 60 BPM "
                      "can be a normal finding.")
    else:
        lines.append("This falls within the normal resting range "
                      "(60–100 BPM).")

    return " ".join(lines)


def _variability_summary(feats: ECGFeatures) -> str:
    """Generate a narrative paragraph about heart-rate variability."""
    sdnn_ms = feats.sdnn * 1000.0
    rmssd_ms = feats.rmssd * 1000.0

    lines = []
    lines.append(f"RR-interval variability metrics: "
                 f"SDNN = {sdnn_ms:.1f} ms, "
                 f"RMSSD = {rmssd_ms:.1f} ms, "
                 f"pNN50 = {feats.pnn50:.1f} %.")

    if sdnn_ms > 160:
        lines.append("The elevated SDNN suggests notable beat-to-beat "
                      "variability.  This could reflect a genuinely "
                      "irregular rhythm (e.g., atrial fibrillation, "
                      "frequent ectopy) or high parasympathetic tone.")
    elif sdnn_ms < 50:
        lines.append("The low SDNN indicates reduced heart-rate "
                      "variability, which may be associated with "
                      "autonomic dysfunction or certain cardiac "
                      "conditions.  Clinical correlation is advised.")
    else:
        lines.append("Variability metrics are within commonly reported "
                      "ranges for short-term recordings.")

    return " ".join(lines)


def _qrs_summary(feats: ECGFeatures) -> str:
    """Generate a narrative paragraph about QRS duration."""
    if feats.qrs_duration_ms is None or len(feats.qrs_duration_ms) == 0:
        return ("QRS duration estimation was not performed or yielded "
                "no results.")

    mean_qrs = float(np.mean(feats.qrs_duration_ms))
    std_qrs = float(np.std(feats.qrs_duration_ms))

    lines = []
    lines.append(f"Estimated mean QRS duration is {mean_qrs:.0f} ms "
                 f"(± {std_qrs:.0f} ms).  Note: this is a rough proxy, "
                 f"not a true QRS delineation measurement.")

    if mean_qrs > 120:
        lines.append("Values above 120 ms may suggest bundle-branch "
                      "block or ventricular conduction delay.  This "
                      "finding should be confirmed with proper QRS "
                      "onset/offset delineation.")
    else:
        lines.append("This is within the normal QRS duration range "
                      "(< 120 ms).")

    return " ".join(lines)


def _flags_narrative(flags: List[AbnormalityFlag]) -> str:
    """Generate a narrative from the rule-engine flags."""
    if not flags:
        return "No rule-based abnormalities were detected in this segment."

    lines = ["The following findings were flagged by the rule engine:"]
    for i, flag in enumerate(flags, 1):
        severity_label = flag.severity.upper()
        lines.append(f"  {i}. [{severity_label}] {flag.name.replace('_', ' ').title()}: "
                     f"{flag.description}")

    lines.append("")
    lines.append("These flags are based on simple threshold rules and "
                 "should be interpreted in clinical context.")
    return "\n".join(lines)


def _eval_narrative(metrics: EvalMetrics) -> str:
    """Generate a narrative about detection performance."""
    lines = []
    lines.append(f"R-peak detection was evaluated against expert "
                 f"annotations with a ±{metrics.tolerance_ms:.0f} ms "
                 f"tolerance window.")
    lines.append(f"Results: Precision = {metrics.precision:.3f}, "
                 f"Recall = {metrics.recall:.3f}, "
                 f"F1-score = {metrics.f1:.3f}.")
    lines.append(f"Matched {metrics.tp} beats correctly, with "
                 f"{metrics.fp} false detections and "
                 f"{metrics.fn} missed beats.")

    if metrics.f1 > 0.95:
        lines.append("Detection quality is excellent.")
    elif metrics.f1 > 0.85:
        lines.append("Detection quality is good, though some beats were "
                      "missed or falsely detected.")
    else:
        lines.append("Detection quality is moderate.  Results should be "
                      "interpreted with caution, and algorithm tuning or "
                      "manual review may be warranted.")

    if abs(metrics.mean_offset_ms) > 20:
        direction = "late" if metrics.mean_offset_ms > 0 else "early"
        lines.append(f"There is a systematic detection bias of "
                     f"{metrics.mean_offset_ms:+.1f} ms ({direction}), "
                     f"which may affect timing-sensitive measurements.")

    return " ".join(lines)


def _disclaimer() -> str:
    """Return the mandatory disclaimer text."""
    return (
        "DISCLAIMER: This analysis is generated by an automated signal "
        "processing pipeline.  It is NOT a clinical diagnosis and must "
        "NOT be used as the sole basis for medical decisions.  All "
        "findings should be reviewed by a qualified healthcare "
        "professional in the context of the patient's full clinical "
        "picture."
    )


# ── Public entry point ─────────────────────────────────────────────────


def generate_explanation(feats: ECGFeatures,
                         rule_results: RuleResults,
                         eval_metrics: EvalMetrics = None,
                         record_id: str = "") -> str:
    """Generate a complete clinical-style narrative report.

    Parameters
    ----------
    feats : ECGFeatures
        Feature set from ``features.extract_features()``.
    rule_results : RuleResults
        Output of ``rules.apply_rules()``.
    eval_metrics : EvalMetrics, optional
        Detection performance metrics (included if provided).
    record_id : str
        Record identifier for the report header.

    Returns
    -------
    str
        Multi-paragraph textual report.
    """
    sections = []

    # Title.
    title = f"ECG Analysis Report — Record {record_id}" if record_id else "ECG Analysis Report"
    sections.append(f"\n{'═' * 50}")
    sections.append(f"  {title}")
    sections.append(f"{'═' * 50}")

    # Heart rate.
    sections.append(_section_header("Heart Rate"))
    sections.append(_rate_summary(feats))

    # Variability.
    sections.append(_section_header("Heart-Rate Variability"))
    sections.append(_variability_summary(feats))

    # QRS duration.
    sections.append(_section_header("QRS Duration (Proxy)"))
    sections.append(_qrs_summary(feats))

    # Abnormality flags.
    sections.append(_section_header("Rule-Based Findings"))
    sections.append(_flags_narrative(rule_results.flags))

    # Detection performance.
    if eval_metrics is not None:
        sections.append(_section_header("Detection Performance"))
        sections.append(_eval_narrative(eval_metrics))

    # Disclaimer.
    sections.append(_section_header("Disclaimer"))
    sections.append(_disclaimer())

    sections.append(f"\n{'═' * 50}\n")

    return "\n".join(sections)


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with dummy data.
    _feats = ECGFeatures(
        rr_intervals=np.array([0.7, 0.65, 0.8, 0.6, 0.75]),
        instantaneous_hr=60.0 / np.array([0.7, 0.65, 0.8, 0.6, 0.75]),
        mean_hr=85.0,
        median_hr=84.0,
        sdnn=0.075,
        rmssd=0.060,
        pnn50=12.0,
        qrs_duration_ms=np.array([90.0, 88.0, 92.0, 91.0, 89.0]),
    )

    from rules import RuleResults
    _rules = RuleResults(flags=[], summary="No abnormalities.", is_normal=True)

    print(generate_explanation(_feats, _rules, record_id="TEST"))
    print("llm_explainer.py OK")
