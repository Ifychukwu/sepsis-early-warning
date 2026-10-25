"""Train and evaluate the sepsis early-warning model against NEWS2 and qSOFA.

Usage:  python -m sepsis.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from sepsis.alerts import burden_curve, evaluate_alerts, subgroup_report, threshold_for_burden  # noqa: E402
from sepsis.data import generate_cohort  # noqa: E402
from sepsis.features import feature_columns, make_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def patient_split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0):
    """Split by patient so no individual appears in both train and test."""
    rng = np.random.default_rng(seed)
    ids = df["patient_id"].unique()
    test_ids = set(rng.choice(ids, int(len(ids) * test_frac), replace=False))
    is_test = df["patient_id"].isin(test_ids)
    return df[~is_test], df[is_test]


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, class_weight={0: 1, 1: 10}, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=30, random_state=0)


def main(n_patients: int = 4000) -> dict:
    df = make_features(generate_cohort(n_patients))
    cols = feature_columns(df)
    train, test = patient_split(df)
    model = build_model().fit(train[cols], train["label"])
    test = test.assign(model_risk=model.predict_proba(test[cols])[:, 1])

    row_metrics = {}
    for name, col in [("model", "model_risk"), ("NEWS2", "news2"), ("qSOFA", "qsofa")]:
        row_metrics[name] = {"auroc": round(roc_auc_score(test["label"], test[col]), 3),
                             "auprc": round(average_precision_score(test["label"], test[col]), 3)}

    # Compare at the false-alarm burden clinicians already accept from NEWS2 >= 5.
    news2_alerts = evaluate_alerts(test, "news2", 5)
    news2_7 = evaluate_alerts(test, "news2", 7)
    qsofa_alerts = evaluate_alerts(test, "qsofa", 2)
    t_matched = threshold_for_burden(test, "model_risk", news2_alerts["false_alarms_per_100_patient_days"])
    model_matched = evaluate_alerts(test, "model_risk", t_matched)
    t_half = threshold_for_burden(test, "model_risk", news2_alerts["false_alarms_per_100_patient_days"] / 2)
    model_half = evaluate_alerts(test, "model_risk", t_half)

    t_7 = threshold_for_burden(test, "model_risk", news2_7["false_alarms_per_100_patient_days"])
    model_7 = evaluate_alerts(test, "model_risk", t_7)

    # Ablation: how much does the model lean on *care-process* signals (when labs
    # were ordered)? These partly encode clinicians' suspicion, so a model that
    # depends on them may just be echoing the team rather than warning it.
    process = [c for c in cols if c.endswith(("_measured", "_hours_since", "_n_draws_24h"))]
    physio_cols = [c for c in cols if c not in process]
    physio_model = build_model().fit(train[physio_cols], train["label"])
    physio_auroc = roc_auc_score(test["label"], physio_model.predict_proba(test[physio_cols])[:, 1])
    physio_auprc = average_precision_score(test["label"], physio_model.predict_proba(test[physio_cols])[:, 1])
    row_metrics["model_without_lab_ordering_features"] = {"auroc": round(physio_auroc, 3), "auprc": round(physio_auprc, 3)}
    test = test.assign(physio_risk=physio_model.predict_proba(test[physio_cols])[:, 1])
    physio_7 = evaluate_alerts(test, "physio_risk",
                               threshold_for_burden(test, "physio_risk", news2_7["false_alarms_per_100_patient_days"]))

    # Subgroup check: does performance hold across age bands and the patient
    # groups that generate most false alarms (COPD, post-operative)?
    groups = {
        "age < 50": test["age"] < 50, "age 50-69": test["age"].between(50, 69), "age >= 70": test["age"] >= 70,
        "COPD": test["copd"] == 1, "no COPD": test["copd"] == 0,
        "post-operative": test["postop"] == 1, "not post-operative": test["postop"] == 0,
    }
    subgroups = subgroup_report(test, groups, {"model (NEWS2 >= 7 alarm rate)": ("model_risk", t_7),
                                               "NEWS2 >= 7": ("news2", 7)})

    imp = permutation_importance(model, test[cols].sample(20000, random_state=0),
                                 test.loc[test[cols].sample(20000, random_state=0).index, "label"],
                                 scoring="average_precision", n_repeats=3, random_state=0)
    top = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(10).round(4)

    results = {
        "data": {"patients": int(df["patient_id"].nunique()), "patient_hours": len(df),
                 "septic_patients_pct": round(100 * df.groupby("patient_id")["sepsis_onset_hour"].first().notna().mean(), 1),
                 "test_patients": int(test["patient_id"].nunique())},
        "row_level": row_metrics,
        "alerting": {
            "NEWS2 >= 5 (NHS urgent response)": news2_alerts,
            "NEWS2 >= 7 (NHS emergency response)": news2_7,
            "qSOFA >= 2": qsofa_alerts,
            "Model, same false-alarm rate as NEWS2 >= 5": model_matched,
            "Model, half the false alarms of NEWS2 >= 5": model_half,
            "Model, same false-alarm rate as NEWS2 >= 7": model_7,
            "Model without lab-ordering features, same false-alarm rate as NEWS2 >= 7": physio_7,
        },
        "subgroups": subgroups,
        "top_features": top.to_dict(),
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": cols, "threshold": t_matched, "threshold_news2_7_rate": t_7},
                ROOT / "models" / "model.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))
    test_ids = test["patient_id"].unique()
    test[test["patient_id"].isin(test_ids[:300])][
        ["patient_id", "hour", "hr", "resp_rate", "temp", "sbp", "spo2", "lactate_last", "news2", "model_risk",
         "sepsis_onset_hour", "label"]].round(4).to_csv(ROOT / "reports" / "test_timelines.csv.gz", index=False)
    _plots(test, top, t_matched)
    return results


def _plots(test: pd.DataFrame, top: pd.Series, t_model: float):
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    m_curve = burden_curve(test, "model_risk", np.unique(np.quantile(test["model_risk"], np.linspace(0.85, 0.999, 30))))
    n_curve = burden_curve(test, "news2", range(3, 11))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(m_curve["false_alarms_per_100_patient_days"], m_curve["sensitivity"], marker="o", ms=3, label="ML model")
    ax.plot(n_curve["false_alarms_per_100_patient_days"], n_curve["sensitivity"], marker="s", ms=4, label="NEWS2 (thresholds 3–10)")
    for _, r in n_curve.iterrows():
        if r["threshold"] in (5, 7):
            ax.annotate(f"NEWS2 ≥ {int(r['threshold'])}", (r["false_alarms_per_100_patient_days"], r["sensitivity"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set(xlabel="False alarms per 100 patient-days", ylabel="Septic patients alerted ≤ 12 h before onset",
           title="Catching sepsis vs alarm fatigue")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / "sensitivity_vs_alarms.png", dpi=130)
    plt.close(fig)

    # One example septic patient timeline.
    septic_ids = test.loc[test["sepsis_onset_hour"].notna(), "patient_id"].unique()
    pid = next(p for p in septic_ids if (test["patient_id"] == p).sum() > 30)
    p = test[test["patient_id"] == pid]
    onset = p["sepsis_onset_hour"].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(p["hour"], p["hr"], label="heart rate"); axes[0].plot(p["hour"], p["resp_rate"] * 4, label="resp rate ×4")
    axes[0].plot(p["hour"], p["sbp"], label="systolic BP"); axes[0].legend(fontsize=7, loc="upper left")
    axes[1].step(p["hour"], p["news2"], where="post", color="tab:orange"); axes[1].axhline(5, ls="--", color="grey", lw=1)
    axes[1].set_ylabel("NEWS2")
    axes[2].plot(p["hour"], p["model_risk"], color="tab:red"); axes[2].axhline(t_model, ls="--", color="grey", lw=1)
    axes[2].set_ylabel("Model risk"); axes[2].set_xlabel("Hours since admission")
    for ax in axes:
        ax.axvline(onset, color="black", lw=1.5); ax.axvspan(onset - 6, onset, color="red", alpha=0.08)
    axes[0].set_title(f"Example patient: sepsis onset at hour {int(onset)} (shaded = 6 h before)")
    fig.tight_layout(); fig.savefig(out / "example_patient.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set(xlabel="Drop in AUPRC when shuffled", title="Most important signals")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
