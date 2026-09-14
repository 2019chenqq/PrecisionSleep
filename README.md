# PrecisionSleep

A lightweight and explainable EEG sleep-staging pipeline designed for future wearable and edge-AI deployment.

PrecisionSleep explores whether single-channel EEG can support efficient and interpretable sleep-stage classification while remaining lightweight enough for future wearable or home-monitoring applications.

---

## Goal

Develop a lightweight sleep-staging pipeline that can classify:

- Wake
- N1
- N2
- N3
- REM

from single-channel EEG using physiologically meaningful features and subject-wise validation.

The current project focuses on building and validating the software pipeline before future integration with wearable hardware.

---

## Current Pipeline

```text
Sleep-EDF PSG
    ↓
Fpz-Cz EEG
    ↓
30-second epochs
    ↓
PSG-grounded hypnogram labels
    ↓
Time-domain & spectral feature extraction
    ↓
15 EEG features
    ↓
90-second causal temporal context
    ↓
Random Forest classifier
    ↓
Wake / N1 / N2 / N3 / REM
    ↓
Leave-One-Subject-Out validation