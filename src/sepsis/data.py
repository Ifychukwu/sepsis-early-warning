"""Synthetic hourly ICU/ward data in the style of the PhysioNet/CinC 2019
sepsis challenge (one row per patient-hour).

What makes it realistic enough to be a fair test:
* vitals are noisy and charted hourly; labs are sparse (every 6–24 h) and
  missing most of the time, so missingness itself is informative;
* about 9% of patients develop sepsis, with a gradual deterioration starting
  6–12 h before onset (heart and breathing rate up, temperature deranged,
  blood pressure down, lactate and white cells abnormal);
* many patients *without* sepsis have abnormal spells too (post-operative
  tachycardia, fever from other causes, COPD patients on oxygen with low
  SpO2), which is what generates false alarms in real hospitals.

To use real data, download the PhysioNet 2019 training set and map its
columns with :func:`load_physionet_psv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VITALS = ["hr", "resp_rate", "temp", "sbp", "map", "spo2"]
LABS = ["wbc", "lactate", "creatinine", "platelets"]


def _patient(pid: int, rng: np.random.Generator, septic: bool) -> pd.DataFrame:
    los = int(rng.integers(24, 160))
    t = np.arange(los)
    age = int(np.clip(rng.normal(64, 15), 18, 95))
    copd = rng.random() < 0.12
    postop = rng.random() < 0.35

    base = {
        "hr": rng.normal(82, 10), "resp_rate": rng.normal(17, 2.5), "temp": rng.normal(36.9, 0.3),
        "sbp": rng.normal(125, 15), "spo2": rng.normal(96.5 if not copd else 92.5, 1.2),
        "wbc": rng.normal(8.5, 2.2), "lactate": abs(rng.normal(1.2, 0.35)), "creatinine": abs(rng.normal(85, 25)),
        "platelets": rng.normal(240, 60),
    }
    drift = {k: np.zeros(los) for k in base}

    # Non-septic abnormal spells (false-alarm generators).
    for _ in range(rng.poisson(2.2 if postop else 1.2)):
        s, d = int(rng.integers(0, los)), int(rng.integers(3, 24))
        kind = rng.choice(["tachy", "fever", "hypotension", "tachypnoea"])
        sl = slice(s, min(los, s + d))
        if kind == "tachy":
            drift["hr"][sl] += rng.uniform(15, 35)
        elif kind == "fever":
            drift["temp"][sl] += rng.uniform(0.8, 1.8)
            drift["hr"][sl] += rng.uniform(5, 15)
        elif kind == "hypotension":
            drift["sbp"][sl] -= rng.uniform(15, 35)
        else:
            drift["resp_rate"][sl] += rng.uniform(4, 9)

    onset = None
    if septic:
        onset = int(rng.integers(12, max(13, los - 4)))
        lead = rng.uniform(3, 12)
        ramp = np.clip((t - (onset - lead)) / lead, 0, None)  # 0 before deterioration, 1 at onset
        ramp = np.minimum(ramp, 1.6)
        sev = rng.uniform(0.25, 1.2)
        # Not every patient shows every sign: sepsis is heterogeneous.
        has = lambda p: float(rng.random() < p)  # noqa: E731
        drift["hr"] += has(0.8) * sev * 25 * ramp
        drift["resp_rate"] += has(0.7) * sev * 7 * ramp
        drift["temp"] += has(0.65) * sev * (1.5 if rng.random() < 0.8 else -1.4) * ramp
        drift["sbp"] -= has(0.5) * sev * 28 * ramp
        drift["spo2"] -= has(0.4) * sev * 3 * ramp
        drift["lactate"] += has(0.6) * sev * 2.2 * ramp
        drift["wbc"] += has(0.65) * sev * (7 if rng.random() < 0.8 else -4) * ramp
        drift["creatinine"] += has(0.4) * sev * 50 * ramp
        drift["platelets"] -= has(0.4) * sev * 70 * ramp

    # Slow physiological wander (random walk) so a patient's "baseline" is a moving target.
    for k, scale in {"hr": 0.6, "resp_rate": 0.2, "temp": 0.03, "sbp": 0.8, "spo2": 0.1}.items():
        drift[k] += np.cumsum(rng.normal(0, scale, los))

    noise = {"hr": 5, "resp_rate": 1.8, "temp": 0.25, "sbp": 8, "spo2": 1.0,
             "wbc": 0.6, "lactate": 0.2, "creatinine": 6, "platelets": 12}
    df = pd.DataFrame({"patient_id": pid, "hour": t, "age": age, "copd": int(copd), "postop": int(postop)})
    for k, b in base.items():
        df[k] = b + drift[k] + rng.normal(0, noise[k], los)
    df["map"] = df["sbp"] * 0.55 + rng.normal(18, 4, los)
    df["spo2"] = df["spo2"].clip(70, 100).round()
    df["on_oxygen"] = ((df["spo2"] < 94) | (copd & (rng.random(los) < 0.7))).astype(int)
    df["alert"] = (rng.random(los) > np.where(t >= (onset or 10**9) - 2, 0.25, 0.01)).astype(int)
    for v in ["hr", "resp_rate", "sbp", "map"]:
        df[v] = df[v].round()
    df["temp"] = df["temp"].round(1)

    # Labs: drawn every 6–24 h (more often once the patient looks unwell); otherwise missing.
    drawn = np.zeros(los, dtype=bool)
    h = 0
    while h < los:
        drawn[h] = True
        h += int(rng.integers(6, 25))
    if septic:
        drawn[max(0, onset - 3): onset + 1] |= rng.random(min(los, onset + 1) - max(0, onset - 3)) < 0.5
    for lab in LABS:
        df[lab] = np.where(drawn, df[lab].round(1), np.nan)
    df["lactate"] = df["lactate"].clip(lower=0.3)

    df["sepsis_onset_hour"] = onset if septic else np.nan
    if septic:
        df = df[df["hour"] <= onset].copy()  # stop once sepsis is recognised
    return df


def generate_cohort(n_patients: int = 4000, sepsis_rate: float = 0.09, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    septic = rng.random(n_patients) < sepsis_rate
    df = pd.concat([_patient(i, rng, s) for i, s in enumerate(septic)], ignore_index=True)
    return add_label(df)


def add_label(df: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Label = 1 if sepsis onset occurs within the next ``horizon`` hours (inclusive of onset hour)."""
    until = df["sepsis_onset_hour"] - df["hour"]
    df["label"] = ((until >= 0) & (until <= horizon)).astype(int)
    return df


def load_physionet_psv(folder: str | Path) -> pd.DataFrame:
    """Load PhysioNet 2019 .psv files into this project's column names."""
    frames = []
    for i, f in enumerate(sorted(Path(folder).glob("*.psv"))):
        d = pd.read_csv(f, sep="|")
        d = d.rename(columns={"HR": "hr", "Resp": "resp_rate", "Temp": "temp", "SBP": "sbp", "MAP": "map",
                              "O2Sat": "spo2", "WBC": "wbc", "Lactate": "lactate", "Creatinine": "creatinine",
                              "Platelets": "platelets", "Age": "age"})
        d["patient_id"], d["hour"] = i, np.arange(len(d))
        onset = d.index[d["SepsisLabel"] == 1]
        d["sepsis_onset_hour"] = (onset[0] + 6) if len(onset) else np.nan  # challenge labels start 6 h early
        d["on_oxygen"], d["alert"], d["copd"], d["postop"] = 0, 1, 0, 0
        frames.append(d)
    return add_label(pd.concat(frames, ignore_index=True))
