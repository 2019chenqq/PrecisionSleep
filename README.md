# PrecisionSleep

Edge-AI EEG sleep staging proof of concept.

## Goal

Develop a lightweight sleep-staging pipeline for wearable or home EEG applications.

## Current Pipeline

Sleep-EDF PSG → EEG → 30 s epochs → EEG features → Random Forest → LOSO validation

## Current Dataset

- 6 subjects
- 8 recordings
- Fpz-Cz EEG
- PSG-grounded hypnogram labels

## Current Best Result

Using 15 EEG features with 90-second causal temporal context:

- Mean Accuracy: 0.7725
- Mean Balanced Accuracy: 0.7015
- Mean Macro F1: 0.6810

## Model Progression

| Version | Macro F1 |
|---|---:|
| 5 spectral features | 0.6133 |
| 15 enhanced features | 0.6508 |
| 45 temporal features | 0.6810 |

## Repository Structure

- `src/` — processing and modeling scripts
- `docs/` — pipeline and experiment documentation
- `results/` — benchmark summaries
- `data/` — local dataset instructions
- `outputs/` — generated intermediate data

## Data

Sleep-EDF Expanded dataset from PhysioNet.

Raw EDF files are not stored in this repository.

## Validation

The current evaluation uses Leave-One-Subject-Out validation to reduce subject leakage.