# Baseline: 5 Spectral Features

## Setup

Features:
- Delta relative power
- Theta relative power
- Alpha relative power
- Sigma relative power
- Beta relative power

Model:
- Random Forest
- 300 estimators
- class_weight = balanced

Validation:
- Leave-One-Subject-Out
- 6 subjects

## Results

- Mean Accuracy: 0.7168
- Mean Balanced Accuracy: 0.6301
- Mean Macro F1: 0.6133
- Macro F1 SD: 0.0447