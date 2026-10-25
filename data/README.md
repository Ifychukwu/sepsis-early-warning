# Data

**The default cohort is synthetic.** Real ICU data (MIMIC, PhysioNet) needs credentialed access and a data use agreement, so `sepsis.data.generate_cohort()` builds a realistic stand-in in memory (seed 3). Nothing is downloaded and no data file is committed.

## What it is

Hourly ward/ICU records in the style of the **PhysioNet/Computing in Cardiology Challenge 2019** (early prediction of sepsis), one row per patient-hour.

| | |
|---|---|
| Patients | 4,000 (1,200 held out for testing) |
| Patient-hours | 346,303 |
| Patients who develop sepsis | 10.4% |
| Length of stay | 24–160 hours; septic patients' records stop at onset |

## How it is generated (`src/sepsis/data.py`)

- **Patients:** age ~ N(64, 15) clipped to 18–95; 12% have COPD (lower baseline SpO2, often on oxygen); 35% are post-operative (more abnormal spells). Each patient has their own baseline for every vital sign and lab.
- **Vitals** (heart rate, respiratory rate, temperature, systolic BP, MAP, SpO2, oxygen, consciousness) are charted hourly with measurement noise and a slow random-walk drift.
- **Labs** (white cell count, lactate, creatinine, platelets) are drawn every 6–24 hours and are missing otherwise; draws become more frequent around sepsis onset, as when clinicians become worried.
- **Sepsis:** onset at a random hour ≥ 12; deterioration starts 3–12 hours before, with random severity. Each sign appears only with some probability (e.g. heart rate 80%, low blood pressure 50%), so sepsis is heterogeneous.
- **False-alarm generators** in everyone: post-operative tachycardia, non-infective fever, transient hypotension, tachypnoea, and COPD patients with low SpO2.

## Schema (main columns)

| Column | Meaning |
|---|---|
| `patient_id`, `hour` | patient and hours since admission |
| `age`, `copd`, `postop` | static patient characteristics |
| `hr`, `resp_rate`, `temp`, `sbp`, `map`, `spo2`, `on_oxygen`, `alert` | hourly vital signs (`alert` = 1 if Alert on ACVPU) |
| `wbc`, `lactate`, `creatinine`, `platelets` | labs, NaN when not drawn |
| `sepsis_onset_hour` | hour of onset, NaN for patients who never develop sepsis |
| `label` | 1 if onset is within the next 6 hours (the training target) |

`make_features()` adds causal rolling statistics, deviations from each patient's baseline, lab missingness and timing, NEWS2 and qSOFA.

## Splits

Patient-level random split, 70% train / 30% test (`patient_split`, seed 0), so no patient appears in both. The model uses 10% of training rows internally for early stopping.

## Using real data

Download the PhysioNet 2019 training sets A and B (`.psv` files, one per patient) from physionet.org and load them with:

```python
from sepsis.data import load_physionet_psv
df = load_physionet_psv("data/physionet2019/training_setA")
```

The loader renames columns to this project's schema and shifts the challenge label back to the estimated onset (the challenge labels start 6 hours early). COPD, post-operative status, oxygen and consciousness are not in the challenge data and are set to defaults, so NEWS2 is approximate on it.

## Licence and git

The synthetic generator is part of this repository. PhysioNet 2019 data is released under the Creative Commons Attribution 4.0 licence on physionet.org; MIMIC requires credentialed access and must never be committed. `data/physionet2019/` and `data/raw/` are git-ignored.
