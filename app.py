"""Sepsis early-warning dashboard: patient timelines, NEWS2 vs ML risk, alert trade-offs.

Run:  python -m sepsis.train  (once, to create reports/)  then  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from sepsis.alerts import alert_hours, evaluate_alerts  # noqa: E402

st.set_page_config(page_title="Sepsis Early Warning", page_icon="🩺", layout="wide")
st.title("🩺 Sepsis Early Warning")
st.caption("Synthetic hospital data · research demo, not a medical device")

timelines_path, metrics_path = ROOT / "reports" / "test_timelines.csv.gz", ROOT / "reports" / "metrics.json"
if not timelines_path.exists():
    st.error("Run `PYTHONPATH=src python -m sepsis.train` first to generate the test-set timelines.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(timelines_path), json.loads(metrics_path.read_text())


df, metrics = load()

with st.sidebar:
    st.header("Alert settings")
    model_t = st.slider("ML risk threshold", 0.01, 0.6, float(round(metrics["alerting"]["Model, same false-alarm rate as NEWS2 >= 7"]["threshold"], 2)), 0.01)
    news_t = st.slider("NEWS2 threshold", 3, 9, 7)
    only_septic = st.toggle("Show septic patients only", True)

tab1, tab2 = st.tabs(["Patient timeline", "Ward-level trade-off"])

with tab1:
    ids = df.loc[df["sepsis_onset_hour"].notna(), "patient_id"].unique() if only_septic else df["patient_id"].unique()
    pid = st.selectbox("Patient", ids)
    p = df[df["patient_id"] == pid]
    onset = p["sepsis_onset_hour"].iloc[0]
    hours = p["hour"].to_numpy()
    m_alerts = alert_hours(hours, p["model_risk"].to_numpy(), model_t)
    n_alerts = alert_hours(hours, p["news2"].to_numpy(), news_t)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sepsis onset", "none" if np.isnan(onset) else f"hour {int(onset)}")
    first = lambda a: next((h for h in a if np.isnan(onset) or onset - 12 <= h <= onset), None)  # noqa: E731
    fm, fn = first(m_alerts), first(n_alerts)
    c2.metric("ML first alert", "—" if fm is None else f"hour {fm}",
              None if (fm is None or np.isnan(onset)) else f"{int(onset - fm)} h warning", delta_color="off")
    c3.metric(f"NEWS2 ≥ {news_t} first alert", "—" if fn is None else f"hour {fn}",
              None if (fn is None or np.isnan(onset)) else f"{int(onset - fn)} h warning", delta_color="off")

    st.markdown("**Vital signs**")
    st.line_chart(p.set_index("hour")[["hr", "resp_rate", "sbp", "spo2"]])
    left, right = st.columns(2)
    left.markdown("**NEWS2**")
    left.line_chart(p.set_index("hour")[["news2"]].assign(threshold=news_t))
    right.markdown("**ML risk (next 6 h)**")
    right.line_chart(p.set_index("hour")[["model_risk"]].assign(threshold=model_t))
    st.caption(f"ML alerts at hours {m_alerts or '—'} · NEWS2 alerts at hours {n_alerts or '—'} (6-hour refractory period)")

with tab2:
    st.markdown("How the settings above would play out across the test patients shown here:")
    rows = {f"ML risk ≥ {model_t:.2f}": evaluate_alerts(df, "model_risk", model_t),
            f"NEWS2 ≥ {news_t}": evaluate_alerts(df, "news2", news_t)}
    st.dataframe(pd.DataFrame(rows).T[["sensitivity", "median_lead_time_h", "false_alarms_per_100_patient_days",
                                      "ppv_of_alerts", "septic_patients"]], use_container_width=True)
    fig = ROOT / "reports" / "figures" / "sensitivity_vs_alarms.png"
    if fig.exists():
        st.image(str(fig), caption="Full test set (1,200 patients)")
