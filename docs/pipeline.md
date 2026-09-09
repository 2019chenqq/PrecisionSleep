# PrecisionSleep Pipeline

```text
Sleep-EDF PSG
    ↓
Fpz-Cz EEG
    ↓
30-second epochs
    ↓
Hypnogram labels
    ↓
Signal feature extraction
    ↓
Wake trimming
    ↓
Subject-aware dataset
    ↓
Leave-One-Subject-Out validation
    ↓
Wake / N1 / N2 / N3 / REM