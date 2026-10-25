# Methods

## 1. Prediction task

Every hour, for every patient, predict whether sepsis onset will occur within the next 6 hours (`label = 1` if 0 ≤ onset − hour ≤ 6). Septic patients' records stop at onset, when sepsis would be recognised and treated. This mirrors the PhysioNet/CinC 2019 challenge set-up.

## 2. Baselines (`src/sepsis/scores.py`)

- **NEWS2** (Royal College of Physicians, 2017): banded points for respiratory rate, SpO2 (scale 1), supplemental oxygen, systolic BP, heart rate, consciousness and temperature. NHS escalation thresholds: ≥ 5 urgent response, ≥ 7 emergency response. Every band is unit-tested against the RCP chart.
- **qSOFA** (Sepsis-3): one point each for respiratory rate ≥ 22, systolic BP ≤ 100 and altered mentation; ≥ 2 flags possible sepsis.

## 3. Features (`src/sepsis/features.py`)

All features use only the current and past hours of the same patient (a unit test changes future values and checks earlier features do not move).

| Group | Features |
|---|---|
| Current vitals | hr, resp_rate, temp, sbp, map, spo2, on_oxygen, alert |
| Vital trends | 6-hour rolling mean, max, min; change over 6 hours; deviation from the patient's own first-6-hour baseline |
| Labs | last observed value (forward-filled) for WBC, lactate, creatinine, platelets |
| Lab process | whether measured this hour, hours since last measurement, draws in the last 24 h |
| Scores | NEWS2, qSOFA, NEWS2 6-hour max and change |
| Derived | shock index (HR / SBP), lactate / SBP |
| Patient | age, COPD, post-operative, hours since admission |

## 4. Model (`src/sepsis/train.py`)

`HistGradientBoostingClassifier`: learning rate 0.05, up to 400 iterations, 31 leaves, ≥ 50 samples per leaf, L2 = 1.0, class weight 10 for positive hours, early stopping on a 10% validation fraction (patience 30), `random_state=0`. Missing labs are handled natively. Split: 70/30 by patient.

## 5. Evaluation

**Row level.** AUROC and AUPRC on all test patient-hours. AUPRC matters more because only a few percent of hours are positive; a random score's AUPRC equals the positive rate.

**Alert level (`src/sepsis/alerts.py`).** Scores are turned into alerts the way a ward system would use them:
- an alert fires when the score reaches the threshold, then is silenced for 6 hours;
- a septic patient is **caught** if an alert fires in the 12 hours up to and including onset; **lead time** = onset − first alert hour in that window;
- **every other alert is a false alarm**, reported per 100 patient-days (alert burden);
- **PPV of alerts** = true alerts / all alerts.

**Equal-burden comparison.** `threshold_for_burden` picks the lowest model threshold whose false-alarm rate does not exceed a target, here NEWS2 ≥ 7's rate, NEWS2 ≥ 5's rate, and half of NEWS2 ≥ 5's rate. Sensitivity and lead time are then compared at (almost) the same alarm burden.

**Ablation.** The model is retrained without lab-process features (`*_measured`, `*_hours_since`, `*_n_draws_24h`), which partly encode clinicians' existing suspicion.

**Subgroups.** `subgroup_report` computes AUROC, AUPRC, sensitivity, lead time and false alarms for age bands (< 50, 50–69, ≥ 70), COPD vs not and post-operative vs not. Thresholds are fixed from the whole test set, as in deployment, so differences in false-alarm rates between groups show up rather than being tuned away.

**Explainability.** Permutation importance (drop in AUPRC, 3 repeats, 20,000 sampled test rows) and per-patient timelines of vitals, NEWS2 and model risk.

## 6. What is not done here

No calibration step (the class-weighted model's probabilities are scores, not risks), no confidence intervals on sensitivity (108 septic test patients, so roughly ±4–9 points), no temporal or external validation, and no sex or ethnicity data in the synthetic cohort.
