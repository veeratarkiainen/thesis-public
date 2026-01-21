# Eye-movement synchronization to cardiac and respiratory cycles (ECG/EOG/RESP/MEG)

**Master’s thesis project** investigating potential synchronization between gaze events (saccades, blinks) and physiological rhythms (cardiac and respiratory cycles) using ECG/EOG/respiration data, with additional MEG analyses.

In the MEG part, the work evaluates **cardiac artifact contribution** to saccade-locked activity at:
- **Sensor level:** evoked responses and RMS/PVE comparisons pre/post cardiac IC removal  
- **Source level:** source estimates (MNE) and region-wise comparison between **cerebellar** and **posterior-cortical** activity (RMS/PVE) with ICA-based cardiac artifact removal

- **Focus:** healthcare time-series analysis, signal preprocessing, event detection, circular statistics, MEG sensor & source-level analysis, independent component analysis (ICA)
- **Stack:** Python, NumPy, pandas, SciPy, MNE-Python, matplotlib, Git  
- **Repo owner:** Veera Tarkiainen

> ⚠️ **Data access & confidentiality:**  
> The raw datasets (especially MEG) are **confidential** and stored in a secure environment. MEG processing scripts were developed to run on the **Aalto Triton cluster (Spyder environment)**, and the corresponding data is accessible **only there**. Therefore, **no raw data is included in this repository**, and MEG results cannot be reproduced outside the secure environment without authorized access. The repository provides the analysis code structure, pipeline logic, and documentation.

---

## What this repo contains

This repository implements an end-to-end analysis workflow with two main components:

### 1) Physiological signals (ECG/EOG/RESP)
1. **Data loading**
2. **Preprocessing & quality checks** (filtering, signal-quality checks)
3. **Event detection & selection** (e.g., saccade/blink detection from EOG)
4. **Feature extraction**
   - Cardiac timing: **R-peak alignment** and **T-wave offset detection** (RTc-based)
   - Respiratory features (phase / rate metrics)
   - Gaze-event timing as a function of cardiac/respiratory phase
5. **Statistics & evaluation**
   - Time-domain analysis (two-way repeated measures ANOVA: Time, Condition, Time×Condition)
   - Circular statistics / phase-preference tests (e.g., Rayleigh, Watson–Williams, Friedman)
   - Post-hoc tests where applicable
6. **Visualization & reporting**

### 2) MEG analysis (Triton / Spyder)
> Runs in the Triton environment due to confidential MEG data access.

1. **Data loading**
2. **Preprocessing & quality checks** (filtering, artifact handling, signal-quality checks)
3. **Event detection & selection** (aligned to gaze events)
4. **Independent Component Analysis (ICA)**
   - Identification of cardiac IC(s)
5. **Sensor-level responses**
   - Evoked responses for: (a) original data, (b) ICA-cleaned data, (c) cardiac IC contribution  
   - RMS and PVE comparisons before/after cardiac IC removal
6. **Source-level responses**
   - Source estimation (MNE inverse solution) using an occipital-restricted sensor space  
   - Source estimates masked to cerebellar segmentation  
   - Comparison of cerebellar vs posterior-cortical activity (RMS/PVE)
7. **Visualization & reporting**

---

## Key contributions (high level)

- Built an **end-to-end Python pipeline** for physiological time-series analysis and a MEG workflow from sensor to source level (MEG/ECG/EOG/RESP)
- Implemented **event detection and selection** with quality control and reproducible parameters
- Applied **statistical testing** (time-domain + circular statistics) to assess synchronization effects and significance
- Produced **clear visual outputs** (phase distributions, sensor/source-level MEG responses, RMS/PVE summaries)

---



