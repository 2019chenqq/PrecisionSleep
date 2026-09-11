from pathlib import Path
import time
import os
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_temporal_context.csv"
)

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv(data_path)

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

print("=" * 60)
print("STAGE 29 - CANONICAL LIGHTWEIGHT MODEL")
print("=" * 60)

print("\nSubjects:")
print(subjects)

print("\nFeature count:")
print(len(feature_columns))


# ============================================================
# 2. Stage 20
# Recording-relative normalization
# ============================================================

normalized_df = df.copy()

for feature in feature_columns:

    normalized_df[feature] = (
        normalized_df
        .groupby("recording")[feature]
        .transform(
            lambda x: (
                x - x.mean()
            ) / (
                x.std() + 1e-8
            )
        )
    )

df = normalized_df

nan_count = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)

print(
    "\nRecording-relative normalization completed."
)

print(
    "NaN count after normalization:",
    nan_count
)


# ============================================================
# 3. Stage 22
# Conservative A-B-A temporal smoothing
# ============================================================

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

    for i in range(
        1,
        len(predictions) - 1
    ):

        prev_stage = predictions[i - 1]
        current_stage = predictions[i]
        next_stage = predictions[i + 1]

        if (
            prev_stage == next_stage
            and current_stage != prev_stage
        ):
            smoothed[i] = prev_stage

    return smoothed


# ============================================================
# 4. Stage 25
# Canonical single-subject demo split
# ============================================================

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


print("\nTest subject:")
print(test_subject)

print(
    "Training epochs:",
    len(X_train)
)

print(
    "Test epochs:",
    len(X_test)
)


# ============================================================
# 5. Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=1,
)

print("\nTraining model...")

train_start = time.perf_counter()

model.fit(
    X_train,
    y_train
)

train_end = time.perf_counter()

training_time_sec = (
    train_end - train_start
)

print(
    "Training time (sec):",
    round(
        training_time_sec,
        4
    )
)


# ============================================================
# 6. Save model + model size
# ============================================================

model_path = (
    results_dir
    / "stage29_lightweight_model.joblib"
)

joblib.dump(
    model,
    model_path
)

model_size_bytes = os.path.getsize(
    model_path
)

model_size_mb = (
    model_size_bytes
    / (1024 * 1024)
)

print(
    "\nModel saved to:"
)

print(
    model_path
)

print(
    "Model size (MB):",
    round(
        model_size_mb,
        3
    )
)


# ============================================================
# Stage 30 - Benchmark Reproducibility
# ============================================================

print("\n" + "=" * 60)
print("STAGE 30 - BENCHMARK REPRODUCIBILITY")
print("=" * 60)

single_epoch = X_test.iloc[[0]]

# Warm-up
for _ in range(20):
    model.predict(single_epoch)

print("\nWarm-up completed.")


# ============================================================
# 1. Single-epoch repeated benchmark
# ============================================================

round_count = 10
repeats_per_round = 100

round_means = []
round_medians = []
round_p95s = []

all_single_epoch_times_ms = []

for round_idx in range(round_count):

    round_times = []

    for _ in range(repeats_per_round):

        start = time.perf_counter()

        model.predict(single_epoch)

        end = time.perf_counter()

        elapsed_ms = (
            end - start
        ) * 1000

        round_times.append(
            elapsed_ms
        )

        all_single_epoch_times_ms.append(
            elapsed_ms
        )

    round_mean = float(
        np.mean(round_times)
    )

    round_median = float(
        np.median(round_times)
    )

    round_p95 = float(
        np.percentile(
            round_times,
            95
        )
    )

    round_means.append(
        round_mean
    )

    round_medians.append(
        round_median
    )

    round_p95s.append(
        round_p95
    )

    print(
        f"Round {round_idx + 1:02d}: "
        f"mean={round_mean:.4f} ms, "
        f"median={round_median:.4f} ms, "
        f"P95={round_p95:.4f} ms"
    )


# ============================================================
# 2. Overall single-epoch statistics
# ============================================================

overall_mean_ms = float(
    np.mean(
        all_single_epoch_times_ms
    )
)

overall_median_ms = float(
    np.median(
        all_single_epoch_times_ms
    )
)

overall_p95_ms = float(
    np.percentile(
        all_single_epoch_times_ms,
        95
    )
)

overall_std_ms = float(
    np.std(
        all_single_epoch_times_ms
    )
)

overall_min_ms = float(
    np.min(
        all_single_epoch_times_ms
    )
)

overall_max_ms = float(
    np.max(
        all_single_epoch_times_ms
    )
)


# ============================================================
# 3. Full-night batch reproducibility
# ============================================================

batch_round_count = 10
batch_times_sec = []

for round_idx in range(batch_round_count):

    start = time.perf_counter()

    model.predict(
        X_test
    )

    end = time.perf_counter()

    elapsed_sec = (
        end - start
    )

    batch_times_sec.append(
        elapsed_sec
    )

    print(
        f"Batch round {round_idx + 1:02d}: "
        f"{elapsed_sec:.6f} sec"
    )


batch_mean_sec = float(
    np.mean(
        batch_times_sec
    )
)

batch_median_sec = float(
    np.median(
        batch_times_sec
    )
)

batch_p95_sec = float(
    np.percentile(
        batch_times_sec,
        95
    )
)

batch_std_sec = float(
    np.std(
        batch_times_sec
    )
)


# ============================================================
# 4. Final summary
# ============================================================

print("\n" + "=" * 60)
print("STAGE 30 SUMMARY")
print("=" * 60)

print("\nSingle-epoch latency")
print("--------------------")

print(
    "Total measurements:",
    len(all_single_epoch_times_ms)
)

print(
    "Mean (ms):",
    round(overall_mean_ms, 4)
)

print(
    "Median (ms):",
    round(overall_median_ms, 4)
)

print(
    "P95 (ms):",
    round(overall_p95_ms, 4)
)

print(
    "Std (ms):",
    round(overall_std_ms, 4)
)

print(
    "Min (ms):",
    round(overall_min_ms, 4)
)

print(
    "Max (ms):",
    round(overall_max_ms, 4)
)


print("\nFull-night batch latency")
print("--------------------")

print(
    "Epoch count:",
    len(X_test)
)

print(
    "Mean (sec):",
    round(batch_mean_sec, 6)
)

print(
    "Median (sec):",
    round(batch_median_sec, 6)
)

print(
    "P95 (sec):",
    round(batch_p95_sec, 6)
)

print(
    "Std (sec):",
    round(batch_std_sec, 6)
)

# ============================================================
# Generate prediction for downstream evaluation
# ============================================================

y_pred_raw = model.predict(
    X_test
)

y_pred_smoothed = smooth_predictions(
    y_pred_raw
)

# ============================================================
# 10. Stage 22 smoothing
# ============================================================

y_pred_smoothed = smooth_predictions(
    y_pred_raw
)


# ============================================================
# 11. Raw + smoothed metrics
# ============================================================

raw_accuracy = accuracy_score(
    y_test,
    y_pred_raw
)

raw_balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        y_pred_raw
    )
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

smoothed_balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        y_pred_smoothed
    )
)

smoothed_macro_f1 = f1_score(
    y_test,
    y_pred_smoothed,
    average="macro"
)


print("\n" + "=" * 60)
print("CANONICAL MODEL PERFORMANCE")
print("=" * 60)

print("\nRaw prediction")

print(
    "Accuracy:",
    round(
        raw_accuracy,
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        raw_balanced_accuracy,
        4
    )
)

print(
    "Macro F1:",
    round(
        raw_macro_f1,
        4
    )
)


print("\nSmoothed prediction")

print(
    "Accuracy:",
    round(
        smoothed_accuracy,
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        smoothed_balanced_accuracy,
        4
    )
)

print(
    "Macro F1:",
    round(
        smoothed_macro_f1,
        4
    )
)

print(
    "\nMacro F1 delta:",
    round(
        smoothed_macro_f1
        - raw_macro_f1,
        4
    )
)


# ============================================================
# 12. Stage 23 sleep summary
# ============================================================

epoch_minutes = 0.5

total_minutes = (
    len(y_pred_smoothed)
    * epoch_minutes
)

stage_order = [
    "N1",
    "N2",
    "N3",
    "REM",
    "Wake",
]

stage_minutes = {}

for stage in stage_order:

    stage_minutes[stage] = (
        (
            y_pred_smoothed
            == stage
        ).sum()
        * epoch_minutes
    )


sleep_minutes = (
    stage_minutes["N1"]
    + stage_minutes["N2"]
    + stage_minutes["N3"]
    + stage_minutes["REM"]
)

wake_minutes = (
    stage_minutes["Wake"]
)

sleep_window_ratio = (
    sleep_minutes
    / total_minutes
    * 100
)

stage_percent = {}

for stage in stage_order:

    stage_percent[stage] = (
        stage_minutes[stage]
        / total_minutes
        * 100
    )


def minutes_to_hm(minutes):

    total = int(
        round(
            minutes
        )
    )

    hours = (
        total // 60
    )

    mins = (
        total % 60
    )

    return (
        f"{hours} h {mins} min"
    )


# ============================================================
# 13. Hypnogram data
# ============================================================

stage_map = {
    "Wake": 0,
    "REM": 1,
    "N1": 2,
    "N2": 3,
    "N3": 4,
}

y_pred_smoothed_num = np.array(
    [
        stage_map[s]
        for s in y_pred_smoothed
    ]
)

epochs = np.arange(
    len(
        y_pred_smoothed
    )
)

hours = (
    epochs
    * 30
    / 3600
)

max_hour = (
    hours.max()
)


# ============================================================
# 14. Stage 25 visual demo report
# ============================================================

fig = plt.figure(
    figsize=(16, 10)
)

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[
        1,
        1.4
    ],
    width_ratios=[
        1.1,
        1
    ],
    hspace=0.35,
    wspace=0.25
)

ax_text = fig.add_subplot(
    gs[0, 0]
)

ax_bar = fig.add_subplot(
    gs[0, 1]
)

ax_hyp = fig.add_subplot(
    gs[1, :]
)

fig.suptitle(
    f"PrecisionSleep Demo Report - {test_subject}",
    fontsize=20,
    y=0.98
)


# ---------- Summary text ----------

ax_text.axis(
    "off"
)

report_text = (
    f"Subject: {test_subject}\n\n"
    f"Analysis window: {minutes_to_hm(total_minutes)}\n"
    f"Predicted sleep time: {minutes_to_hm(sleep_minutes)}\n"
    f"Predicted wake time: {minutes_to_hm(wake_minutes)}\n"
    f"Sleep within analysis window: "
    f"{sleep_window_ratio:.1f}%\n\n"
    f"Model performance (smoothed)\n"
    f"Accuracy: {smoothed_accuracy:.3f}\n"
    f"Balanced Accuracy: "
    f"{smoothed_balanced_accuracy:.3f}\n"
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


# ---------- Stage distribution ----------

bar_labels = [
    "N1",
    "N2",
    "N3",
    "REM",
    "Wake"
]

bar_values = [
    stage_percent[s]
    for s in bar_labels
]

ax_bar.barh(
    bar_labels,
    bar_values
)

ax_bar.set_title(
    "Sleep stage distribution"
)

ax_bar.set_xlabel(
    "Percent of analysis window (%)"
)

ax_bar.set_xlim(
    0,
    max(bar_values) + 10
)

ax_bar.grid(
    True,
    axis="x",
    alpha=0.3
)

for i, value in enumerate(
    bar_values
):

    ax_bar.text(
        value + 0.5,
        i,
        f"{value:.1f}%",
        va="center",
        fontsize=11
    )


# ---------- Hypnogram ----------

ax_hyp.step(
    hours,
    y_pred_smoothed_num,
    where="post",
    linewidth=2
)

ax_hyp.set_yticks(
    [
        0,
        1,
        2,
        3,
        4
    ]
)

ax_hyp.set_yticklabels(
    [
        "Wake",
        "REM",
        "N1",
        "N2",
        "N3"
    ]
)

ax_hyp.invert_yaxis()

ax_hyp.set_title(
    "Smoothed Predicted Hypnogram"
)

ax_hyp.set_xlabel(
    "Time (hours)"
)

ax_hyp.set_ylabel(
    "Sleep stage"
)

ax_hyp.grid(
    True,
    alpha=0.3
)

ax_hyp.set_xticks(
    np.arange(
        0,
        np.ceil(
            max_hour
        ) + 1,
        1
    )
)


# ---------- Footer ----------

fig.text(
    0.01,
    0.01,
    (
        "Prototype output for algorithm development. "
        "Not a clinical diagnosis."
    ),
    fontsize=10
)


report_path = (
    results_dir
    / f"stage29_demo_report_{test_subject}.png"
)

plt.savefig(
    report_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved visual report to:"
)

print(
    report_path
)


# ============================================================
# 15. Save benchmark CSV
# ============================================================

benchmark_df = pd.DataFrame([
    {
        "subject": test_subject,
        "tree_count": model.n_estimators,
        "single_epoch_measurements":
            len(all_single_epoch_times_ms),

        "single_epoch_mean_ms":
            overall_mean_ms,

        "single_epoch_median_ms":
            overall_median_ms,

        "single_epoch_p95_ms":
            overall_p95_ms,

        "single_epoch_std_ms":
            overall_std_ms,

        "single_epoch_min_ms":
            overall_min_ms,

        "single_epoch_max_ms":
            overall_max_ms,

        "batch_rounds":
            batch_round_count,

        "batch_mean_sec":
            batch_mean_sec,

        "batch_median_sec":
            batch_median_sec,

        "batch_p95_sec":
            batch_p95_sec,

        "batch_std_sec":
            batch_std_sec,
    }
])

output_path = (
    results_dir
    / "stage30_benchmark_reproducibility.csv"
)

benchmark_df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)