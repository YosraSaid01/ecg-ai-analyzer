"""
rules.py — Rule-Based Abnormality Detection

Applies clinically motivated thresholds to the features computed in
``features.py`` and returns structured flags with human-readable
explanations.

This module does NOT make diagnoses.  It flags patterns that a
clinician would want to review.  All thresholds are based on widely
accepted clinical cut-offs (AHA / ESC guidelines) and can be
overridden by the caller.

Detected patterns:
    • Tachycardia         — mean HR > 100 BPM
    • Bradycardia         — mean HR < 60 BPM
    • Irregular rhythm    — elevated RR-interval variability
    • Premature beats     — individual RR intervals significantly
                            shorter than the local average
    • Wide QRS            — estimated QRS duration > 120 ms
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from features import ECGFeatures


# ── Threshold defaults ─────────────────────────────────────────────────

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "tachycardia_bpm": 100.0,
    "bradycardia_bpm": 60.0,
    "irregular_sdnn_sec": 0.16,       # SDNN > 160 ms suggests irregularity
    "irregular_rmssd_sec": 0.10,      # supportive criterion
    "premature_rr_ratio": 0.75,       # beat < 75 % of local mean RR
    "wide_qrs_ms": 120.0,             # QRS ≥ 120 ms (bundle-branch block range)
}


@dataclass
class AbnormalityFlag:
    """A single detected abnormality.

    Attributes
    ----------
    name : str
        Short machine-readable label (e.g. "tachycardia").
    severity : str
        One of "info", "mild", "moderate", "significant".
    description : str
        One-sentence human-readable explanation.
    value : float
        The numeric metric that triggered the flag.
    threshold : float
        The threshold that was exceeded.
    """
    name: str
    severity: str
    description: str
    value: float
    threshold: float


@dataclass
class RuleResults:
    """Aggregated output of the rule engine.

    Attributes
    ----------
    flags : List[AbnormalityFlag]
        All triggered abnormality flags, possibly empty.
    summary : str
        One-paragraph textual summary suitable for display.
    is_normal : bool
        True if no flags were raised.
    """
    flags: List[AbnormalityFlag] = field(default_factory=list)
    summary: str = ""
    is_normal: bool = True


# ── Individual rule functions ──────────────────────────────────────────


def check_tachycardia(feats: ECGFeatures,
                      threshold: float = None) -> AbnormalityFlag | None:
    """Flag if mean heart rate exceeds the tachycardia threshold.

    Parameters
    ----------
    feats : ECGFeatures
        Computed features.
    threshold : float, optional
        BPM cut-off (default 100).

    Returns
    -------
    AbnormalityFlag or None
    """
    thr = threshold or DEFAULT_THRESHOLDS["tachycardia_bpm"]
    if feats.mean_hr > thr:
        severity = "moderate" if feats.mean_hr > 120 else "mild"
        return AbnormalityFlag(
            name="tachycardia",
            severity=severity,
            description=(
                f"Mean heart rate is {feats.mean_hr:.0f} BPM, "
                f"above the {thr:.0f} BPM threshold for tachycardia."
            ),
            value=feats.mean_hr,
            threshold=thr,
        )
    return None


def check_bradycardia(feats: ECGFeatures,
                      threshold: float = None) -> AbnormalityFlag | None:
    """Flag if mean heart rate is below the bradycardia threshold.

    Parameters
    ----------
    feats : ECGFeatures
        Computed features.
    threshold : float, optional
        BPM cut-off (default 60).

    Returns
    -------
    AbnormalityFlag or None
    """
    thr = threshold or DEFAULT_THRESHOLDS["bradycardia_bpm"]
    if feats.mean_hr < thr and feats.mean_hr > 0:
        severity = "moderate" if feats.mean_hr < 50 else "mild"
        return AbnormalityFlag(
            name="bradycardia",
            severity=severity,
            description=(
                f"Mean heart rate is {feats.mean_hr:.0f} BPM, "
                f"below the {thr:.0f} BPM threshold for bradycardia."
            ),
            value=feats.mean_hr,
            threshold=thr,
        )
    return None


def check_irregular_rhythm(feats: ECGFeatures,
                           sdnn_thr: float = None,
                           rmssd_thr: float = None) -> AbnormalityFlag | None:
    """Flag if RR-interval variability suggests an irregular rhythm.

    Uses SDNN as the primary criterion and RMSSD as a supportive check.

    Parameters
    ----------
    feats : ECGFeatures
        Computed features.
    sdnn_thr : float, optional
        SDNN threshold in seconds (default 0.16).
    rmssd_thr : float, optional
        RMSSD threshold in seconds (default 0.10).

    Returns
    -------
    AbnormalityFlag or None
    """
    s_thr = sdnn_thr or DEFAULT_THRESHOLDS["irregular_sdnn_sec"]
    r_thr = rmssd_thr or DEFAULT_THRESHOLDS["irregular_rmssd_sec"]

    if feats.sdnn > s_thr:
        both = feats.rmssd > r_thr
        severity = "moderate" if both else "mild"
        return AbnormalityFlag(
            name="irregular_rhythm",
            severity=severity,
            description=(
                f"RR-interval variability is elevated "
                f"(SDNN = {feats.sdnn * 1000:.0f} ms, "
                f"RMSSD = {feats.rmssd * 1000:.0f} ms). "
                f"This may indicate an irregular or variable rhythm."
            ),
            value=feats.sdnn,
            threshold=s_thr,
        )
    return None


def check_premature_beats(feats: ECGFeatures,
                          ratio_thr: float = None) -> AbnormalityFlag | None:
    """Flag if any RR intervals are significantly shorter than average.

    A beat arriving much earlier than expected may indicate a premature
    atrial or ventricular contraction.

    Parameters
    ----------
    feats : ECGFeatures
        Computed features.
    ratio_thr : float, optional
        An RR interval shorter than ``ratio_thr × mean_RR`` triggers
        the flag (default 0.75).

    Returns
    -------
    AbnormalityFlag or None
    """
    thr = ratio_thr or DEFAULT_THRESHOLDS["premature_rr_ratio"]
    rr = feats.rr_intervals
    if len(rr) < 3:
        return None

    mean_rr = np.mean(rr)
    short_count = int(np.sum(rr < thr * mean_rr))

    if short_count > 0:
        pct = 100.0 * short_count / len(rr)
        severity = "moderate" if pct > 5.0 else "mild"
        return AbnormalityFlag(
            name="premature_beats",
            severity=severity,
            description=(
                f"{short_count} beat(s) ({pct:.1f} %) arrived significantly "
                f"earlier than expected (RR < {thr * 100:.0f} % of mean). "
                f"Possible premature contractions."
            ),
            value=float(short_count),
            threshold=thr,
        )
    return None


def check_wide_qrs(feats: ECGFeatures,
                   threshold_ms: float = None) -> AbnormalityFlag | None:
    """Flag if estimated QRS duration exceeds the wide-QRS threshold.

    A QRS ≥ 120 ms may indicate bundle-branch block or ventricular
    origin.  Note: this uses a rough proxy, not true QRS delineation.

    Parameters
    ----------
    feats : ECGFeatures
        Computed features (must include qrs_duration_ms).
    threshold_ms : float, optional
        QRS width threshold in ms (default 120).

    Returns
    -------
    AbnormalityFlag or None
    """
    thr = threshold_ms or DEFAULT_THRESHOLDS["wide_qrs_ms"]
    if feats.qrs_duration_ms is None or len(feats.qrs_duration_ms) == 0:
        return None

    mean_qrs = float(np.mean(feats.qrs_duration_ms))
    if mean_qrs > thr:
        severity = "moderate" if mean_qrs > 150 else "mild"
        return AbnormalityFlag(
            name="wide_qrs",
            severity=severity,
            description=(
                f"Mean estimated QRS duration is {mean_qrs:.0f} ms, "
                f"above the {thr:.0f} ms threshold. "
                f"This is a rough estimate and may warrant further review."
            ),
            value=mean_qrs,
            threshold=thr,
        )
    return None


# ── Aggregate rule engine ──────────────────────────────────────────────


def apply_rules(feats: ECGFeatures,
                thresholds: Dict[str, float] = None) -> RuleResults:
    """Run all rule checks and return aggregated results.

    Parameters
    ----------
    feats : ECGFeatures
        Feature set from ``features.extract_features()``.
    thresholds : dict, optional
        Override individual thresholds by key (see DEFAULT_THRESHOLDS).

    Returns
    -------
    RuleResults
        Aggregated flags and summary.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    checks = [
        check_tachycardia(feats, thr["tachycardia_bpm"]),
        check_bradycardia(feats, thr["bradycardia_bpm"]),
        check_irregular_rhythm(feats, thr["irregular_sdnn_sec"],
                               thr["irregular_rmssd_sec"]),
        check_premature_beats(feats, thr["premature_rr_ratio"]),
        check_wide_qrs(feats, thr["wide_qrs_ms"]),
    ]

    flags = [f for f in checks if f is not None]

    if not flags:
        summary = (
            f"No rule-based abnormalities detected.  "
            f"Mean heart rate is {feats.mean_hr:.0f} BPM with "
            f"SDNN = {feats.sdnn * 1000:.0f} ms.  "
            f"The rhythm appears regular within normal limits."
        )
    else:
        parts = [f.description for f in flags]
        summary = "  ".join(parts)

    return RuleResults(
        flags=flags,
        summary=summary,
        is_normal=(len(flags) == 0),
    )


# ── Standalone test ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate features for a mildly tachycardic, irregular record.
    _rr = np.array([0.55, 0.52, 0.60, 0.48, 0.58, 0.53, 0.61, 0.50,
                     0.57, 0.54, 0.59, 0.51, 0.56, 0.62, 0.49])
    _ihr = 60.0 / _rr

    _feats = ECGFeatures(
        rr_intervals=_rr,
        instantaneous_hr=_ihr,
        mean_hr=float(np.mean(_ihr)),
        median_hr=float(np.median(_ihr)),
        sdnn=float(np.std(_rr, ddof=1)),
        rmssd=float(np.sqrt(np.mean(np.diff(_rr) ** 2))),
        pnn50=float(100.0 * np.sum(np.abs(np.diff(_rr)) > 0.05) / len(np.diff(_rr))),
        qrs_duration_ms=np.full(len(_rr), 95.0),
    )

    _results = apply_rules(_feats)
    print(f"Normal: {_results.is_normal}")
    for _f in _results.flags:
        print(f"  [{_f.severity}] {_f.name}: {_f.description}")
    print(f"\nSummary: {_results.summary}")
    print("rules.py OK")
