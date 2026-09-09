# Experiments

## Experiment 1 — Single-subject baseline

Random epoch split within one subject.

Result:
- Accuracy ≈ 0.839
- Macro F1 ≈ 0.775

Limitation:
Possible subject and temporal leakage.

---

## Experiment 2 — Subject-wise validation

Training subjects and test subject are separated.

Result:
- 3-subject LOSO Macro F1 ≈ 0.585

---

## Experiment 3 — 6-subject baseline

Five spectral features.

Result:
- Macro F1 = 0.6133

---

## Experiment 4 — Enhanced EEG features

15 EEG features.

Result:
- Macro F1 = 0.6508

---

## Experiment 5 — Causal temporal context

90-second context using previous two epochs and the current epoch.

Result:
- Macro F1 = 0.6810