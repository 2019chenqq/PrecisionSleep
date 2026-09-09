# Temporal Context: 45 Features

## Setup

A causal 90-second context was used:

- previous 60 seconds
- current 30-second epoch

Each epoch contains 15 EEG features.

Total model input:
- 45 features

Model:
- Random Forest
- 300 estimators
- class_weight = balanced

Validation:
- Leave-One-Subject-Out
- 6 subjects

## Results

- Mean Accuracy: 0.7725
- Mean Balanced Accuracy: 0.7015
- Mean Macro F1: 0.6810
- Macro F1 SD: 0.0742

## Progression

| Version | Macro F1 |
|---|---:|
| 5 spectral features | 0.6133 |
| 15 enhanced features | 0.6508 |
| 45 temporal features | 0.6810 |