"""Clinical early-warning scores used as baselines.

NEWS2 (Royal College of Physicians, 2017) is the standard early-warning
score in NHS hospitals; a total of 5+ triggers an urgent clinical response.
qSOFA (Sepsis-3) flags suspected sepsis when 2+ of: respiratory rate >= 22,
systolic BP <= 100, altered mentation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _band(x, edges, points):
    """Score ``x`` by bands: edges are the upper bounds (inclusive) of each band."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, points[-1], dtype=float)
    for edge, pts in reversed(list(zip(edges, points[:-1]))):
        out = np.where(x <= edge, pts, out)
    return np.where(np.isnan(x), 0, out)


def news2(df: pd.DataFrame) -> np.ndarray:
    """NEWS2 total (SpO2 scale 1). Expects resp_rate, spo2, on_oxygen, sbp, hr, temp, alert (1 = Alert)."""
    rr = _band(df["resp_rate"], [8, 11, 20, 24], [3, 1, 0, 2, 3])
    spo2 = _band(df["spo2"], [91, 93, 95], [3, 2, 1, 0])
    o2 = np.where(df["on_oxygen"] == 1, 2, 0)
    sbp = _band(df["sbp"], [90, 100, 110, 219], [3, 2, 1, 0, 3])
    hr = _band(df["hr"], [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    cons = np.where(df["alert"] == 1, 0, 3)
    temp = _band(df["temp"], [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    return rr + spo2 + o2 + sbp + hr + cons + temp


def qsofa(df: pd.DataFrame) -> np.ndarray:
    return ((df["resp_rate"] >= 22).astype(int) + (df["sbp"] <= 100).astype(int) + (df["alert"] == 0).astype(int)).to_numpy()
