# Model card: sepsis early-warning model

| | |
|---|---|
| **Model** | scikit-learn `HistGradientBoostingClassifier` (class weight 10 for positive hours, early stopping) on 64 causal features from hourly vitals, sparse labs, NEWS2 and qSOFA |
| **Task** | Every hour, score the risk that sepsis onset occurs within the next 6 hours; alert when the score crosses a threshold (6-hour refractory period) |
| **Training data** | 2,800 synthetic patients (70% of a 4,000-patient cohort, 346,303 patient-hours overall, 10.4% develop sepsis). See [data/README.md](../data/README.md) |
| **Evaluation data** | 1,200 held-out synthetic patients, 108 of them septic |
| **Saved model** | `models/model.joblib` (~1.5 MB): model, feature list, and two thresholds (0.0451 = NEWS2 ≥ 5's alarm rate; 0.1243 = NEWS2 ≥ 7's alarm rate) |

## Intended use
A research and portfolio prototype showing how to build and, above all, **evaluate** an early-warning model against NEWS2 the way a ward would experience it. In any future clinical use it would be decision support: it prompts a nurse or doctor to review a patient and never diagnoses sepsis or triggers treatment on its own.

## Out of scope
- Any use on real patients. It has not been trained or validated on real data.
- Replacing NEWS2 or local escalation policy. NEWS2 is the national standard in NHS hospitals and must continue to run.
- Paediatric or maternity patients (NEWS2 itself does not apply), emergency department triage, or outpatient settings.
- Diagnosing sepsis or deciding on antibiotics.

## Regulatory note (UK)
Software that predicts deterioration to inform clinical decisions for individual patients is likely to be a **medical device** (software as a medical device, SaMD) under the UK Medical Devices Regulations 2002, regulated by the **MHRA**. Before clinical use it would need UKCA (or CE) marking under the appropriate risk class, a quality management system, clinical evaluation, and post-market surveillance; the MHRA's Software and AI as a Medical Device guidance applies. NHS deployment also requires clinical risk management under **DCB0129** (manufacturer) and **DCB0160** (deploying organisation), a DPIA under UK GDPR, and evidence to the NICE Evidence Standards Framework for digital health technologies. None of this has been done here.

## Performance (test set, all from `reports/metrics.json`)

**Row level** (every patient-hour):

| Score | ROC AUC | PR AUC |
|---|---|---|
| **ML model** | **0.819** | **0.369** |
| ML model without lab-ordering features | 0.806 | 0.298 |
| NEWS2 | 0.709 | 0.042 |
| qSOFA | 0.643 | 0.025 |

**Alerting** (108 septic patients; caught = alert in the 12 h before onset):

| Rule | Septic patients alerted | Median lead time | False alarms / 100 patient-days | PPV of alerts |
|---|---|---|---|---|
| qSOFA ≥ 2 | 50.9% | 2 h | 20.1 | 5.9% |
| NEWS2 ≥ 7 | 59.3% | 2 h | 33.5 | 4.2% |
| **Model at NEWS2 ≥ 7's alarm rate** | **96.3%** | **3 h** | **31.9** | **7.0%** |
| Model without lab-ordering features, same rate | 91.7% | 3 h | 32.3 | 6.6% |
| NEWS2 ≥ 5 | 80.6% | 5 h | 97.6 | 2.0% |
| Model at NEWS2 ≥ 5's alarm rate | 99.1% | 5 h | 97.3 | 2.5% |
| Model at half NEWS2 ≥ 5's alarm rate | 97.2% | 4 h | 48.8 | 4.7% |

With 108 septic patients, each sensitivity has a 95% interval of roughly ±4 points (near 96%) to ±9 points (near 60%).

**Subgroups** (model at NEWS2 ≥ 7's overall alarm rate, threshold fixed for everyone):

| Subgroup | Patients (septic) | Model ROC AUC | Model PR AUC | Model alerted | Model false alarms /100 pd | NEWS2 ≥ 7 alerted | NEWS2 ≥ 7 false alarms /100 pd |
|---|---|---|---|---|---|---|---|
| Age < 50 | 205 (24) | 0.814 | 0.380 | 91.7% | 19.6 | 62.5% | 36.9 |
| Age 50–69 | 583 (54) | 0.836 | 0.420 | 98.1% | 32.5 | 57.4% | 32.0 |
| Age ≥ 70 | 412 (30) | 0.800 | 0.287 | 96.7% | 37.0 | 60.0% | 34.0 |
| COPD | 137 (19) | 0.678 | 0.300 | 94.7% | 22.7 | 94.7% | **168.3** |
| No COPD | 1,063 (89) | 0.855 | 0.387 | 96.6% | 33.1 | 51.7% | 16.5 |
| Post-operative | 443 (42) | 0.808 | 0.368 | 97.6% | 29.6 | 76.2% | 38.6 |
| Not post-operative | 757 (66) | 0.826 | 0.370 | 95.5% | 33.2 | 48.5% | 30.6 |

- **COPD:** NEWS2 (scored on SpO2 scale 1 for everyone here) produces 168 false alarms per 100 patient-days in COPD patients, ten times the rate in others. The model avoids that, but its row-level ROC AUC is much lower for COPD (0.678 vs 0.855): it separates septic from non-septic hours less well in a group whose baseline physiology is abnormal. Only 19 septic COPD patients are in the test set.
- **Older patients** get more false alarms from the model (37.0 vs 19.6 per 100 patient-days for under-50s) and a lower PR AUC (0.287), so alarm fatigue would fall unevenly across wards.
- Sex and ethnicity are not in the synthetic data and could not be checked (see below).

**Most important signals** (permutation importance, drop in PR AUC): last lactate (0.057), hours since last white cell count (0.053), white cell draws in 24 h (0.036), heart rate vs the patient's baseline (0.016), hours since admission (0.015).

## Limitations
- **Synthetic data.** Results demonstrate the method, not clinical performance. Real ICU data is messier; the best PhysioNet 2019 entries reached around 0.8 ROC AUC with a much harder label.
- **Care-process features.** Lab-ordering timing partly encodes clinicians' suspicion. The physiology-only model keeps most of the gain (91.7% vs 96.3% alerted), but in real data this shortcut could be larger.
- **Label definition.** Sepsis-3 onset times in real records are uncertain, and here septic records stop at onset.
- **Hours since admission** is a feature; in a different case mix or length of stay this can shift.
- **Scores are not calibrated probabilities** (class weighting), so thresholds must be chosen by alarm burden, as done here.
- **No temporal or external validation**, and small subgroup sample sizes.

## Ethical considerations
- **Alarm fatigue is a safety risk in its own right.** That is why every comparison is made at equal false-alarm rates and alert burden is reported per subgroup.
- **Missing protected characteristics.** The synthetic cohort has no sex or ethnicity. Pulse oximeters can over-estimate oxygen saturation in people with darker skin (reviewed by the UK's independent review of equity in medical devices, 2024), which affects both NEWS2 and this model. A real validation must report performance by sex, ethnicity, age and deprivation.
- **Automation bias.** Staff may trust a silent model and miss deterioration it does not flag. The model should add to NEWS2, never suppress it.
- **Privacy.** Real patient-hour data is special category health data under UK GDPR; it requires a lawful basis, a DPIA and secure, access-controlled processing. No real data is in this repository.
- **Human oversight.** Every alert goes to a clinician for assessment; the app shows the vitals, NEWS2 and model risk side by side so the reason for an alert can be checked.
