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
print("STAGE 26 - LIGHTWEIGHT BENCHMARK")
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

# ============================================================
# Stage 27 - Tree Reduction Experiment
# ============================================================

tree_candidates = [
    300,
    200,
    150,
    100,
    50,
]

benchmark_rows = []

print("\n" + "=" * 60)
print("STAGE 27 - TREE REDUCTION EXPERIMENT")
print("=" * 60)

single_epoch = X_test.iloc[[0]]

for tree_count in tree_candidates:

    print("\n" + "-" * 60)
    print(f"Testing {tree_count} trees")
    print("-" * 60)

    model = RandomForestClassifier(
        n_estimators=tree_count,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )

    # -------------------------
    # Training
    # -------------------------

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_end = time.perf_counter()

    training_time_sec = (
        train_end - train_start
    )


    # -------------------------
    # Save model
    # -------------------------

    model_path = (
        results_dir
        / f"stage27_rf_{tree_count}_trees.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    model_size_mb = (
        os.path.getsize(model_path)
        / (1024 * 1024)
    )


    # -------------------------
    # Warm-up
    # -------------------------

    model.predict(
        single_epoch
    )


    # -------------------------
    # Full-night batch inference
    # -------------------------

    batch_start = time.perf_counter()

    y_pred_raw = model.predict(
        X_test
    )

    batch_end = time.perf_counter()

    fullnight_inference_sec = (
        batch_end - batch_start
    )

    avg_batch_epoch_ms = (
        fullnight_inference_sec
        / len(X_test)
        * 1000
    )


    # -------------------------
    # Single epoch latency
    # -------------------------

    repeat_count = 100
    single_epoch_times_ms = []

    for _ in range(repeat_count):

        start = time.perf_counter()

        model.predict(
            single_epoch
        )

        end = time.perf_counter()

        single_epoch_times_ms.append(
            (end - start) * 1000
        )

    single_epoch_avg_ms = float(
        np.mean(single_epoch_times_ms)
    )

    single_epoch_median_ms = float(
        np.median(single_epoch_times_ms)
    )

    single_epoch_p95_ms = float(
        np.percentile(
            single_epoch_times_ms,
            95
        )
    )


    # -------------------------
    # Smoothing
    # -------------------------

    y_pred_smoothed = smooth_predictions(
        y_pred_raw
    )


    # -------------------------
    # Metrics
    # -------------------------

    raw_macro_f1 = f1_score(
        y_test,
        y_pred_raw,
        average="macro"
    )

    smoothed_macro_f1 = f1_score(
        y_test,
        y_pred_smoothed,
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


    # -------------------------
    # Save row
    # -------------------------

    benchmark_rows.append({
        "trees": tree_count,
        "model_size_mb": model_size_mb,
        "training_time_sec": training_time_sec,
        "fullnight_inference_sec": fullnight_inference_sec,
        "batch_epoch_ms": avg_batch_epoch_ms,
        "single_epoch_avg_ms": single_epoch_avg_ms,
        "single_epoch_median_ms": single_epoch_median_ms,
        "single_epoch_p95_ms": single_epoch_p95_ms,
        "raw_macro_f1": raw_macro_f1,
        "smoothed_macro_f1": smoothed_macro_f1,
        "smoothed_accuracy": smoothed_accuracy,
        "smoothed_balanced_accuracy": smoothed_balanced_accuracy,
    })


    # -------------------------
    # Print current result
    # -------------------------

    print(
        "Model size (MB):",
        round(model_size_mb, 3)
    )

    print(
        "Single epoch avg (ms):",
        round(single_epoch_avg_ms, 4)
    )

    print(
        "Single epoch P95 (ms):",
        round(single_epoch_p95_ms, 4)
    )

    print(
        "Raw Macro F1:",
        round(raw_macro_f1, 4)
    )

    print(
        "Smoothed Macro F1:",
        round(smoothed_macro_f1, 4)
    )


# ============================================================
# Summary table
# ============================================================

benchmark_df = pd.DataFrame(
    benchmark_rows
)

benchmark_df = benchmark_df.sort_values(
    "trees",
    ascending=False
)

print("\n" + "=" * 60)
print("STAGE 27 SUMMARY")
print("=" * 60)

print(
    benchmark_df[
        [
            "trees",
            "model_size_mb",
            "single_epoch_avg_ms",
            "single_epoch_p95_ms",
            "smoothed_macro_f1",
            "smoothed_accuracy",
            "smoothed_balanced_accuracy",
        ]
    ].round(4)
)


# ============================================================
# Save CSV
# ============================================================

csv_path = (
    results_dir
    / "stage27_tree_reduction_results.csv"
)

benchmark_df.to_csv(
    csv_path,
    index=False
)

print("\nSaved results to:")
print(csv_path)


# ============================================================
# Find compact candidate
# ============================================================

baseline_f1 = benchmark_df.loc[
    benchmark_df["trees"] == 300,
    "smoothed_macro_f1"
].iloc[0]

benchmark_df["f1_drop_vs_300"] = (
    baseline_f1
    - benchmark_df["smoothed_macro_f1"]
)

print("\nF1 drop compared with 300 trees:")

print(
    benchmark_df[
        [
            "trees",
            "model_size_mb",
            "smoothed_macro_f1",
            "f1_drop_vs_300",
        ]
    ].round(4)
)

print("\nStage 27 completed successfully.")