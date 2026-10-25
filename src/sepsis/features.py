"""Causal (past-only) features for hourly prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sepsis.data import LABS, VITALS
from sepsis.scores import news2, qsofa


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "hour"]).copy()
    g = df.groupby("patient_id", sort=False)

    for lab in LABS:
        df[f"{lab}_measured"] = df[lab].notna().astype(int)
        df[f"{lab}_last"] = g[lab].ffill()
        last_hour = df["hour"].where(df[lab].notna())
        df[f"{lab}_hours_since"] = df["hour"] - last_hour.groupby(df["patient_id"]).ffill()
        df[f"{lab}_n_draws_24h"] = g[f"{lab}_measured"].transform(lambda s: s.rolling(24, min_periods=1).sum())

    for v in VITALS:
        roll = g[v]
        df[f"{v}_mean6"] = roll.transform(lambda s: s.rolling(6, min_periods=1).mean())
        df[f"{v}_max6"] = roll.transform(lambda s: s.rolling(6, min_periods=1).max())
        df[f"{v}_min6"] = roll.transform(lambda s: s.rolling(6, min_periods=1).min())
        df[f"{v}_delta6"] = df[v] - roll.shift(6)
        # Deviation from the patient's own first-6-hour baseline.
        baseline = roll.transform(lambda s: s.expanding().mean().where(s.index < s.index[0] + 6).ffill())
        df[f"{v}_vs_baseline"] = df[v] - baseline

    df["news2"] = news2(df)
    df["qsofa"] = qsofa(df)
    df["news2_max6"] = df.groupby("patient_id", sort=False)["news2"].transform(lambda s: s.rolling(6, min_periods=1).max())
    df["news2_delta6"] = df["news2"] - df.groupby("patient_id", sort=False)["news2"].shift(6)
    df["shock_index"] = df["hr"] / df["sbp"].clip(lower=40)
    df["lactate_x_sbp"] = df["lactate_last"] / df["sbp"].clip(lower=40)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"patient_id", "sepsis_onset_hour", "label", *LABS}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
