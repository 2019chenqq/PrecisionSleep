from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_temporal_context.csv"
)

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)

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

subjects = sorted(df["subject"].unique())

print("==============================")
print("STAGE 24 - VISUAL SLEEP REPORT")
print("==============================")

print("\nSubjects:")
print(subjects)

print("\nFeature count:")
print(len(feature_columns))


# ==========================================
# 2. Stage 20 normalization
# ==========================================

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

print("\nRecording-relative normalization completed.")
print(
    "NaN count after normalization:",
    df[feature_columns].isna().sum().sum()
)


# ==========================================
# 3. Stage 22 smoothing function
# ==========================================

def smooth_predictions(predictions):
    """
    Conservative A-B-A smoothing.

    Example:
    N2, Wake, N2
    ->
    N2, N2, N2

    Only isolated single-epoch jumps are corrected.
    """

    smoothed = predictions.copy()

    for i in range(1, len(predictions) - 1):

        prev_stage = predictions[i - 1]
        current_stage = predictions[i]
        next_stage = predictions[i + 1]

        if (
            prev_stage == next_stage
            and current_stage != prev_stage
        ):
            smoothed[i] = prev_stage

    return smoothed


# ==========================================
# 4. 單一 subject 測試
# ==========================================

test_subject = "SC402"

train_subjects = [
    s
    for s in subjects
    if s != test_subject
]

train_df = df[
    df["subject"].isin(train_subjects)
].copy()

test_df = df[
    df["subject"] == test_subject
].copy()

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

y_pred_raw = model.predict(
    X_test
)

y_pred_smoothed = smooth_predictions(
    y_pred_raw
)


# ============================================================
# Stage 25 - Canonical Demo Metrics
# ============================================================

raw_accuracy = accuracy_score(
    y_test,
    y_pred_raw
)

raw_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred_raw
)

raw_macro_f1 = f1_score(
    y_test,
    y_pred_raw,
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
print("STAGE 25 - CANONICAL DEMO PIPELINE")
print("=" * 60)

print("\nSubject:")
print(test_subject)

print("\nRaw prediction")
print("--------------------")
print(
    "Accuracy:",
    round(raw_accuracy, 4)
)
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

print("\nImprovement")
print("--------------------")
print(
    "Macro F1 delta:",
    round(
        smoothed_macro_f1 - raw_macro_f1,
        4
    )
)

# ==========================================
# 6. Stage 23 summary variables
# ==========================================

epoch_minutes = 0.5
total_minutes = len(y_pred_smoothed) * epoch_minutes

stage_order = ["N1", "N2", "N3", "REM", "Wake"]

stage_minutes = {}
for stage in stage_order:
    stage_minutes[stage] = (
        (y_pred_smoothed == stage).sum() * epoch_minutes
    )

sleep_minutes = (
    stage_minutes["N1"]
    + stage_minutes["N2"]
    + stage_minutes["N3"]
    + stage_minutes["REM"]
)

wake_minutes = stage_minutes["Wake"]

sleep_window_ratio = (
    sleep_minutes / total_minutes * 100
)

stage_percent = {}
for stage in stage_order:
    stage_percent[stage] = (
        stage_minutes[stage] / total_minutes * 100
    )


def minutes_to_hm(minutes):
    total = int(round(minutes))
    hours = total // 60
    mins = total % 60
    return f"{hours} h {mins} min"


# ==========================================
# 7. Hypnogram data
# ==========================================

stage_map = {
    "Wake": 0,
    "REM": 1,
    "N1": 2,
    "N2": 3,
    "N3": 4,
}

y_pred_smoothed_num = np.array(
    [stage_map[s] for s in y_pred_smoothed]
)

epochs = np.arange(len(y_pred_smoothed))
hours = epochs * 30 / 3600
max_hour = hours.max()


# ==========================================
# 8. Stage 24 - Visual Sleep Report
# ==========================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1, 1.4],
    width_ratios=[1.1, 1],
    hspace=0.35,
    wspace=0.25
)

ax_text = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])
ax_hyp = fig.add_subplot(gs[1, :])

fig.suptitle(
    f"PrecisionSleep Demo Report - {test_subject}",
    fontsize=20,
    y=0.98
)

# ---------- 左上：summary text ----------
ax_text.axis("off")

report_text = (
    f"Subject: {test_subject}\n\n"
    f"Analysis window: {minutes_to_hm(total_minutes)}\n"
    f"Predicted sleep time: {minutes_to_hm(sleep_minutes)}\n"
    f"Predicted wake time: {minutes_to_hm(wake_minutes)}\n"
    f"Sleep within analysis window: {sleep_window_ratio:.1f}%\n\n"
    f"Model performance (smoothed)\n"
    f"Accuracy: {smoothed_accuracy:.3f}\n"
    f"Balanced Accuracy: {smoothed_balanced_accuracy:.3f}\n"
    f"Macro F1: {smoothed_macro_f1:.3f}\n"
)

ax_text.text(
    0.00,
    1.00,
    report_text,
    va="top",
    ha="left",
    fontsize=14,
    linespacing=1.6
)


# ---------- 右上：stage distribution ----------
bar_labels = ["N1", "N2", "N3", "REM", "Wake"]
bar_values = [stage_percent[s] for s in bar_labels]

ax_bar.barh(
    bar_labels,
    bar_values
)

ax_bar.set_title("Sleep stage distribution")
ax_bar.set_xlabel("Percent of analysis window (%)")
ax_bar.set_xlim(0, max(bar_values) + 10)
ax_bar.grid(True, axis="x", alpha=0.3)

for i, v in enumerate(bar_values):
    ax_bar.text(
        v + 0.5,
        i,
        f"{v:.1f}%",
        va="center",
        fontsize=11
    )


# ---------- 下方：smoothed hypnogram ----------
ax_hyp.step(
    hours,
    y_pred_smoothed_num,
    where="post",
    linewidth=2
)

ax_hyp.set_yticks([0, 1, 2, 3, 4])
ax_hyp.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])
ax_hyp.invert_yaxis()

ax_hyp.set_title("Smoothed Predicted Hypnogram")
ax_hyp.set_xlabel("Time (hours)")
ax_hyp.set_ylabel("Sleep stage")
ax_hyp.grid(True, alpha=0.3)

ax_hyp.set_xticks(
    np.arange(
        0,
        np.ceil(max_hour) + 1,
        1
    )
)


# ---------- footer ----------
fig.text(
    0.01,
    0.01,
    "Prototype output for algorithm development. Not a clinical diagnosis.",
    fontsize=10
)

output_path = (
    results_dir
    / f"stage25_demo_report_{test_subject}.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved visual report to:")
print(output_path)

metrics_df = pd.DataFrame([
    {
        "subject": test_subject,
        "raw_accuracy": raw_accuracy,
        "raw_balanced_accuracy": raw_balanced_accuracy,
        "raw_macro_f1": raw_macro_f1,
        "smoothed_accuracy": smoothed_accuracy,
        "smoothed_balanced_accuracy": smoothed_balanced_accuracy,
        "smoothed_macro_f1": smoothed_macro_f1,
        "analysis_minutes": total_minutes,
        "predicted_sleep_minutes": sleep_minutes,
        "predicted_wake_minutes": wake_minutes,
        "sleep_window_percent": sleep_window_ratio,
        "N1_percent": stage_percent["N1"],
        "N2_percent": stage_percent["N2"],
        "N3_percent": stage_percent["N3"],
        "REM_percent": stage_percent["REM"],
        "Wake_percent": stage_percent["Wake"],
    }
])

metrics_path = (
    results_dir
    / f"stage25_demo_metrics_{test_subject}.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print("\nSaved demo metrics to:")
print(metrics_path)