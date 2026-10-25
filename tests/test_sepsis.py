import numpy as np
import pandas as pd
import pytest

from sepsis.alerts import alert_hours, evaluate_alerts, subgroup_report, threshold_for_burden
from sepsis.data import add_label, generate_cohort
from sepsis.features import feature_columns, make_features
from sepsis.scores import news2, qsofa


@pytest.fixture(scope="module")
def cohort():
    return make_features(generate_cohort(n_patients=300, seed=5))


def vitals(**kw):
    base = dict(resp_rate=16, spo2=97, on_oxygen=0, sbp=120, hr=70, alert=1, temp=37.0)
    base.update(kw)
    return pd.DataFrame([base])


@pytest.mark.parametrize("change,points", [
    ({}, 0),
    ({"resp_rate": 8}, 3), ({"resp_rate": 10}, 1), ({"resp_rate": 22}, 2), ({"resp_rate": 25}, 3),
    ({"spo2": 91}, 3), ({"spo2": 93}, 2), ({"spo2": 95}, 1),
    ({"on_oxygen": 1}, 2),
    ({"sbp": 90}, 3), ({"sbp": 95}, 2), ({"sbp": 105}, 1), ({"sbp": 220}, 3),
    ({"hr": 40}, 3), ({"hr": 45}, 1), ({"hr": 100}, 1), ({"hr": 120}, 2), ({"hr": 131}, 3),
    ({"alert": 0}, 3),
    ({"temp": 35.0}, 3), ({"temp": 35.5}, 1), ({"temp": 38.5}, 1), ({"temp": 39.1}, 2),
])
def test_news2_bands_match_rcp_chart(change, points):
    assert news2(vitals(**change))[0] == points


def test_qsofa():
    assert qsofa(vitals(resp_rate=24, sbp=95, alert=0))[0] == 3
    assert qsofa(vitals())[0] == 0


def test_cohort_shape_and_labels(cohort):
    rate = cohort.groupby("patient_id")["sepsis_onset_hour"].first().notna().mean()
    assert 0.03 < rate < 0.2
    pos = cohort[cohort["label"] == 1]
    assert ((pos["sepsis_onset_hour"] - pos["hour"]).between(0, 6)).all()
    # Septic patients' records stop at onset.
    assert (cohort["hour"] <= cohort["sepsis_onset_hour"].fillna(np.inf)).all()


def test_label_horizon():
    df = add_label(pd.DataFrame({"hour": range(10), "sepsis_onset_hour": 8.0}), horizon=3)
    assert df["label"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]


def test_features_are_causal(cohort):
    """Changing a future hour must not change any feature at an earlier hour."""
    pid = cohort["patient_id"].iloc[0]
    raw = generate_cohort(n_patients=300, seed=5)
    altered = raw.copy()
    rows = altered.index[altered["patient_id"] == pid]
    altered.loc[rows[-1], ["hr", "lactate"]] = [250, 15.0]
    a, b = make_features(raw), make_features(altered)
    cols = feature_columns(a)
    early = a.index[(a["patient_id"] == pid) & (a["hour"] < a.loc[rows[-1], "hour"])]
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])


def test_alert_refractory_period():
    hours = np.arange(20)
    scores = np.ones(20)
    assert alert_hours(hours, scores, 0.5, refractory=6) == [0, 6, 12, 18]


def test_evaluate_alerts_counts():
    df = pd.DataFrame({
        "patient_id": [1] * 5 + [2] * 5, "hour": list(range(5)) * 2,
        "sepsis_onset_hour": [4.0] * 5 + [np.nan] * 5,
        "score": [0, 0, 0.9, 0.9, 0.9, 0, 0.9, 0, 0, 0],
    })
    out = evaluate_alerts(df, "score", 0.5)
    assert out["sensitivity"] == 1.0 and out["median_lead_time_h"] == 2.0
    assert out["false_alarms_per_100_patient_days"] == pytest.approx(100 * 1 / (10 / 24), rel=1e-3)


def test_threshold_for_burden_respects_target(cohort):
    t = threshold_for_burden(cohort, "news2", 50.0, candidates=np.arange(3, 12))
    assert evaluate_alerts(cohort, "news2", t)["false_alarms_per_100_patient_days"] <= 50.0


def test_subgroup_report_partitions_patients_and_matches_overall(cohort):
    groups = {"COPD": cohort["copd"] == 1, "no COPD": cohort["copd"] == 0}
    rep = subgroup_report(cohort, groups, {"NEWS2 >= 5": ("news2", 5)})
    assert rep["COPD"]["patients"] + rep["no COPD"]["patients"] == cohort["patient_id"].nunique()
    septic = cohort.groupby("patient_id")["sepsis_onset_hour"].first().notna().sum()
    assert rep["COPD"]["septic_patients"] + rep["no COPD"]["septic_patients"] == septic
    # Alerting on the whole cohort equals the subgroup-level results combined.
    overall = evaluate_alerts(cohort, "news2", 5)
    caught = sum(rep[g]["NEWS2 >= 5"]["sensitivity"] * rep[g]["septic_patients"] for g in groups)
    assert caught / septic == pytest.approx(overall["sensitivity"], abs=0.01)
    r = rep["no COPD"]["NEWS2 >= 5"]
    assert 0.5 <= r["auroc"] <= 1 and 0 < r["auprc"] <= 1 and r["false_alarms_per_100_patient_days"] >= 0
