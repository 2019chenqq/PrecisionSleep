from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_temporal_context.csv"
)


df = pd.read_csv(data_path)


# ==========================================
# 1. 自動找出 45 個 temporal features
# ==========================================

metadata_columns = [
    "subject",
    "recording",
    "epoch",
    "start_sec",
    "label",
]


feature_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]


subjects = sorted(
    df["subject"].unique()
)


print("==============================")
print("TEMPORAL LOSO VALIDATION")
print("==============================")

print("\nSubjects:")
print(subjects)

print("\nFeature count:")
print(len(feature_columns))


results = []

# ============================================================
# Stage 20 - Subject / Recording Relative Normalization
# ============================================================

normalized_df = df.copy()

for feature in feature_columns:

    normalized_df[feature] = (
        normalized_df
        .groupby("recording")[feature]
        .transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    )

df = normalized_df

print("\nSubject-relative normalization completed.")

print(
    "NaN count after normalization:",
    df[feature_columns].isna().sum().sum()
)

# ==========================================
# 2. LOSO
# ==========================================

test_subject = "SC402"
train_subjects = [
    s
    for s in subjects
    if s != test_subject
]


train_df = df[
    df["subject"].isin(
        train_subjects
    )
].copy()

test_df = df[
    df["subject"]
    == test_subject
].copy()


X_train = train_df[
    feature_columns
]

y_train = train_df[
    "label"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "label"
]


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)

# ============================================================
# Stage 22 - Temporal smoothing
# ============================================================

y_pred_smoothed = y_pred.copy()

for i in range(1, len(y_pred) - 1):
    prev_stage = y_pred[i - 1]
    current_stage = y_pred[i]
    next_stage = y_pred[i + 1]

    if (
        prev_stage == next_stage
        and current_stage != prev_stage
    ):
        y_pred_smoothed[i] = prev_stage

# ============================================================
# Stage 23 - Sleep Report Summary
# ============================================================

epoch_minutes = 0.5

stage_counts = pd.Series(
    y_pred_smoothed
).value_counts()

wake_epochs = stage_counts.get("Wake", 0)
n1_epochs = stage_counts.get("N1", 0)
n2_epochs = stage_counts.get("N2", 0)
n3_epochs = stage_counts.get("N3", 0)
rem_epochs = stage_counts.get("REM", 0)

total_epochs = len(y_pred_smoothed)

total_minutes = total_epochs * epoch_minutes

wake_minutes = wake_epochs * epoch_minutes

sleep_minutes = (
    n1_epochs
    + n2_epochs
    + n3_epochs
    + rem_epochs
) * epoch_minutes

n1_percent = n1_epochs / total_epochs * 100
n2_percent = n2_epochs / total_epochs * 100
n3_percent = n3_epochs / total_epochs * 100
rem_percent = rem_epochs / total_epochs * 100
wake_percent = wake_epochs / total_epochs * 100

sleep_window_ratio = (
    sleep_minutes
    / total_minutes
    * 100
)

# ============================================================
# Stage 22 - Hypnogram Comparison
# ============================================================

stage_map = {
    "Wake": 0,
    "REM": 1,
    "N1": 2,
    "N2": 3,
    "N3": 4
}

y_true_num = np.array(
    [stage_map[s] for s in y_test]
)

y_pred_num = np.array(
    [stage_map[s] for s in y_pred]
)

epochs = np.arange(len(y_test))

# 每個 epoch = 30 秒
hours = epochs * 30 / 3600

max_hour = hours.max()

fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 10),
    sharex=True
)

# Ground Truth
axes[0].step(
    hours,
    y_true_num,
    where="post"
)

axes[0].set_yticks(
    [0, 1, 2, 3, 4]
)

axes[0].set_yticklabels(
    ["Wake", "REM", "N1", "N2", "N3"]
)

axes[0].set_ylabel("Sleep stage")
axes[0].set_title(
    f"Ground Truth Hypnogram - {test_subject}"
)

axes[0].grid(
    True,
    alpha=0.3
)


# Prediction
axes[1].set_xticks(
    np.arange(
        0,
        np.ceil(max_hour) + 1,
        1
    )
)

# Raw Prediction
axes[1].step(
    hours,
    y_pred_num,
    where="post"
)

axes[1].set_yticks(
    [0, 1, 2, 3, 4]
)

axes[1].set_yticklabels(
    ["Wake", "REM", "N1", "N2", "N3"]
)

axes[1].set_ylabel("Sleep stage")

axes[1].set_title(
    f"Raw Prediction - {test_subject}"
)

axes[1].grid(
    True,
    alpha=0.3
)

axes[1].invert_yaxis()

# ============================================================
# Smoothed Prediction
# ============================================================

y_pred_smoothed_num = np.array(
    [stage_map[s] for s in y_pred_smoothed]
)

axes[2].step(
    hours,
    y_pred_smoothed_num,
    where="post"
)

axes[2].set_yticks(
    [0, 1, 2, 3, 4]
)

axes[2].set_yticklabels(
    ["Wake", "REM", "N1", "N2", "N3"]
)

axes[2].set_xlabel("Time (hours)")
axes[2].set_ylabel("Sleep stage")

axes[2].set_title(
    f"Smoothed Prediction - {test_subject}"
)

axes[2].grid(
    True,
    alpha=0.3
)

axes[2].invert_yaxis()

axes[2].set_xticks(
    np.arange(
        0,
        np.ceil(max_hour) + 1,
        1
    )
)

plt.tight_layout()

plt.savefig(
    f"results/stage22_smoothing_{test_subject}.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Stage 22 - Raw vs Smoothed Evaluation
# ============================================================

raw_accuracy = accuracy_score(
    y_test,
    y_pred
)

raw_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

raw_macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


smoothed_accuracy = accuracy_score(
    y_test,
    y_pred_smoothed
)

smoothed_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred_smoothed
)

smoothed_macro_f1 = f1_score(
    y_test,
    y_pred_smoothed,
    average="macro"
)


print("\n" + "=" * 60)
print("STAGE 22 - TEMPORAL SMOOTHING COMPARISON")
print("=" * 60)

print("\nTest subject:")
print(test_subject)


print("\nRaw prediction")
print("--------------------")
print("Accuracy:", round(raw_accuracy, 4))
print(
    "Balanced Accuracy:",
    round(raw_balanced_accuracy, 4)
)
print(
    "Macro F1:",
    round(raw_macro_f1, 4)
)


print("\nSmoothed prediction")
print("--------------------")
print(
    "Accuracy:",
    round(smoothed_accuracy, 4)
)
print(
    "Balanced Accuracy:",
    round(smoothed_balanced_accuracy, 4)
)
print(
    "Macro F1:",
    round(smoothed_macro_f1, 4)
)


print("\nChange")
print("--------------------")
print(
    "Accuracy delta:",
    round(
        smoothed_accuracy - raw_accuracy,
        4
    )
)

print(
    "Balanced Accuracy delta:",
    round(
        smoothed_balanced_accuracy
        - raw_balanced_accuracy,
        4
    )
)

print(
    "Macro F1 delta:",
    round(
        smoothed_macro_f1
        - raw_macro_f1,
        4
    )
)


print("\nRaw Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0
    )
)


print("\nSmoothed Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_smoothed,
        digits=3,
        zero_division=0
    )
)

print("\n" + "=" * 60)
print("STAGE 23 - SLEEP REPORT")
print("=" * 60)

print("Subject:", test_subject)

print(
    "Recording window (min):",
    round(total_minutes, 1)
)

print(
    "Predicted sleep time (min):",
    round(sleep_minutes, 1)
)

print(
    "Predicted wake time (min):",
    round(wake_minutes, 1)
)

print(
    "Sleep within analysis window (%):",
    round(sleep_window_ratio, 1)
)

print("\nSleep stage distribution:")

print(
    "N1 (%):",
    round(n1_percent, 1)
)

print(
    "N2 (%):",
    round(n2_percent, 1)
)

print(
    "N3 / Deep sleep (%):",
    round(n3_percent, 1)
)

print(
    "REM (%):",
    round(rem_percent, 1)
)

print(
    "Wake (%):",
    round(wake_percent, 1)
)