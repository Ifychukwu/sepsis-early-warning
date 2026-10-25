"""Turn hourly risk scores into alerts and measure what clinicians would experience.

Row-level AUROC is not how an early-warning system is judged on the ward.
What matters is: how many septic patients get an alert *in time*, how many
hours of warning they get, and how many false alarms nurses must respond to.

Alert rules
* An alert fires when the score crosses the threshold, then is silenced for
  ``refractory`` hours (as real systems do to avoid repeated pages).
* A septic patient is *detected* if an alert fires in the 12 hours before
  onset. Lead time = onset hour - first alert hour in that window.
* Every other alert is a false alarm (including alerts long before onset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 12


def alert_hours(hours: np.ndarray, scores: np.ndarray, threshold: float, refractory: int = 6) -> list[int]:
    fired, last = [], -10**9
    above = scores >= threshold
    for h, a in zip(hours, above):
        if a and h - last >= refractory:
            fired.append(int(h))
            last = h
    return fired


def evaluate_alerts(df: pd.DataFrame, score_col: str, threshold: float, refractory: int = 6) -> dict:
    detected, leads, false_alarms, septic = 0, [], 0, 0
    patient_hours = len(df)
    for _, p in df.groupby("patient_id", sort=False):
        fired = alert_hours(p["hour"].to_numpy(), p[score_col].to_numpy(), threshold, refractory)
        onset = p["sepsis_onset_hour"].iloc[0]
        if np.isnan(onset):
            false_alarms += len(fired)
            continue
        septic += 1
        in_window = [h for h in fired if onset - WINDOW <= h <= onset]
        false_alarms += len(fired) - len(in_window)
        if in_window:
            detected += 1
            leads.append(onset - in_window[0])
    sens = detected / max(septic, 1)
    alerts_total = detected + false_alarms
    return {
        "threshold": round(float(threshold), 4),
        "sensitivity": round(sens, 3),
        "median_lead_time_h": float(np.median(leads)) if leads else 0.0,
        "false_alarms_per_100_patient_days": round(100 * false_alarms / (patient_hours / 24), 1),
        "ppv_of_alerts": round(detected / max(alerts_total, 1), 3),
        "septic_patients": septic,
    }


def threshold_for_burden(df: pd.DataFrame, score_col: str, target_fa_per_100pd: float,
                         candidates: np.ndarray | None = None) -> float:
    """Smallest threshold whose false-alarm rate does not exceed the target."""
    if candidates is None:
        candidates = np.unique(np.quantile(df[score_col], np.linspace(0.80, 0.999, 120)))
    best = float(candidates.max())
    for t in sorted(candidates, reverse=True):
        if evaluate_alerts(df, score_col, t)["false_alarms_per_100_patient_days"] <= target_fa_per_100pd:
            best = float(t)
        else:
            break
    return best


def burden_curve(df: pd.DataFrame, score_col: str, thresholds) -> pd.DataFrame:
    return pd.DataFrame([evaluate_alerts(df, score_col, t) for t in thresholds])


def subgroup_report(df: pd.DataFrame, groups: dict, rules: dict) -> dict:
    """Row-level discrimination and ward-level alerting per patient subgroup.

    ``groups`` maps a subgroup name to a boolean row mask (constant within a
    patient); ``rules`` maps a rule name to ``(score_col, threshold)``. Thresholds
    are fixed from the whole test set, as they would be in deployment, so the
    false-alarm rate is allowed to differ between subgroups.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {}
    for name, mask in groups.items():
        d = df[mask]
        row = {"patients": int(d["patient_id"].nunique()),
               "septic_patients": int(d.groupby("patient_id")["sepsis_onset_hour"].first().notna().sum())}
        for rule, (col, thr) in rules.items():
            both = d["label"].nunique() == 2
            a = evaluate_alerts(d, col, thr)
            row[rule] = {
                "auroc": round(float(roc_auc_score(d["label"], d[col])), 3) if both else None,
                "auprc": round(float(average_precision_score(d["label"], d[col])), 3) if both else None,
                "sensitivity": a["sensitivity"],
                "median_lead_time_h": a["median_lead_time_h"],
                "false_alarms_per_100_patient_days": a["false_alarms_per_100_patient_days"],
            }
        out[name] = row
    return out
