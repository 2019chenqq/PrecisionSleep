from pathlib import Path
import time

import numpy as np
import pandas as pd

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
# Stage 32
# Real-time normalization strategy benchmark
#
# Goal:
# Compare multiple deployable normalization strategies
# against the Stage 29/30 offline recording-relative baseline.
#
# Canonical model:
# RandomForestClassifier
# n_estimators = 150
#
# Test subject:
# SC402
# ============================================================


print("=" * 78)
print("STAGE 32 - REAL-TIME NORMALIZATION STRATEGY BENCHMARK")
print("=" * 78)


# ============================================================
# 1. Load dataset
# ============================================================

df_raw = pd.read_csv(data_path)

metadata_columns = [
    "subject",
    "recording",
    "epoch",
    "start_sec",
    "label",
]

feature_columns = [
    col
    for col in df_raw.columns
    if col not in metadata_columns
]

subjects = sorted(
    df_raw["subject"].unique()
)

test_subject = "SC402"

print("\nTest subject:")
print(test_subject)

print("\nSubjects:")
print(subjects)

print("\nFeature count:")
print(len(feature_columns))


# ============================================================
# 2. Train / test split
# ============================================================

train_raw = df_raw[
    df_raw["subject"] != test_subject
].copy()

test_raw = df_raw[
    df_raw["subject"] == test_subject
].copy()

print("\nTraining epochs:")
print(len(train_raw))

print("\nTest epochs:")
print(len(test_raw))


# ============================================================
# 3. Normalization strategies
# ============================================================

def offline_recording_normalize(
    input_df,
    features,
):
    """
    Stage 29/30 baseline.

    Uses mean/std from the entire recording.
    This is NOT causal because future epochs contribute
    to the normalization statistics.
    """

    output_df = input_df.copy()

    for feature in features:

        output_df[feature] = (
            output_df
            .groupby("recording")[feature]
            .transform(
                lambda x: (
                    x - x.mean()
                ) / (
                    x.std() + 1e-8
                )
            )
        )

    return output_df


def causal_expanding_normalize(
    input_df,
    features,
):
    """
    Stage 31 causal expanding normalization.

    Each epoch uses only the current and previous epochs
    from the same recording.
    """

    output_df = input_df.copy()

    for feature in features:

        normalized_values = pd.Series(
            index=output_df.index,
            dtype=float,
        )

        for recording, group in output_df.groupby(
            "recording",
            sort=False,
        ):

            values = group[
                feature
            ].astype(float)

            expanding_mean = (
                values
                .expanding(
                    min_periods=1
                )
                .mean()
            )

            expanding_std = (
                values
                .expanding(
                    min_periods=2
                )
                .std()
            )

            normalized = (
                values
                - expanding_mean
            ) / (
                expanding_std
                + 1e-8
            )

            normalized = (
                normalized
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
            )

            normalized_values.loc[
                group.index
            ] = normalized

        output_df[feature] = (
            normalized_values
            .loc[output_df.index]
            .values
        )

    return output_df


def rolling_window_normalize(
    input_df,
    features,
    window_epochs,
):
    """
    Causal rolling-window normalization.

    Includes the current epoch plus previous epochs.
    No future information is used.

    window_epochs:
        10  = 5 minutes
        20  = 10 minutes
        60  = 30 minutes
        because each epoch = 30 seconds.
    """

    output_df = input_df.copy()

    for feature in features:

        normalized_values = pd.Series(
            index=output_df.index,
            dtype=float,
        )

        for recording, group in output_df.groupby(
            "recording",
            sort=False,
        ):

            values = group[
                feature
            ].astype(float)

            rolling_mean = (
                values
                .rolling(
                    window=window_epochs,
                    min_periods=1,
                )
                .mean()
            )

            rolling_std = (
                values
                .rolling(
                    window=window_epochs,
                    min_periods=2,
                )
                .std()
            )

            normalized = (
                values
                - rolling_mean
            ) / (
                rolling_std
                + 1e-8
            )

            normalized = (
                normalized
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
            )

            normalized_values.loc[
                group.index
            ] = normalized

        output_df[feature] = (
            normalized_values
            .loc[output_df.index]
            .values
        )

    return output_df


def warmup_baseline_normalize(
    input_df,
    features,
    warmup_epochs,
):
    """
    Fixed baseline normalization using only the first
    N epochs of each recording.

    After baseline statistics are established,
    the same mean/std are used for the rest of
    the recording.

    This is deployable AFTER the warm-up period.

    warmup_epochs:
        10 = 5 minutes
        20 = 10 minutes
        60 = 30 minutes
    """

    output_df = input_df.copy()

    for recording, group in output_df.groupby(
        "recording",
        sort=False,
    ):

        warmup_group = group.iloc[
            :warmup_epochs
        ]

        for feature in features:

            baseline_mean = float(
                warmup_group[
                    feature
                ].mean()
            )

            baseline_std = float(
                warmup_group[
                    feature
                ].std()
            )

            if (
                np.isnan(baseline_std)
                or baseline_std < 1e-8
            ):
                baseline_std = 1.0

            normalized = (
                group[feature].astype(float)
                - baseline_mean
            ) / (
                baseline_std
                + 1e-8
            )

            normalized = (
                normalized
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
            )

            output_df.loc[
                group.index,
                feature,
            ] = normalized.values

    return output_df


def training_global_statistics(
    train_df,
    features,
):
    """
    Calculate fixed global mean/std from training data only.

    This can be packaged with the model and used at inference
    without needing future subject data.
    """

    means = {}
    stds = {}

    for feature in features:

        mean_value = float(
            train_df[
                feature
            ].mean()
        )

        std_value = float(
            train_df[
                feature
            ].std()
        )

        if (
            np.isnan(std_value)
            or std_value < 1e-8
        ):
            std_value = 1.0

        means[feature] = mean_value
        stds[feature] = std_value

    return means, stds


def fixed_statistics_normalize(
    input_df,
    features,
    means,
    stds,
):
    """
    Normalize using fixed statistics calculated
    from the training set only.
    """

    output_df = input_df.copy()

    for feature in features:

        output_df[feature] = (
            output_df[feature].astype(float)
            - means[feature]
        ) / (
            stds[feature]
            + 1e-8
        )

    output_df[features] = (
        output_df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return output_df


# ============================================================
# 4. Smoothing
# ============================================================

def smooth_predictions(
    predictions,
):
    """
    Conservative A-B-A smoothing
    from Stage 22 / 29 / 30 / 31.
    """

    smoothed = predictions.copy()

    for i in range(
        1,
        len(predictions) - 1,
    ):

        prev_stage = predictions[
            i - 1
        ]

        current_stage = predictions[
            i
        ]

        next_stage = predictions[
            i + 1
        ]

        if (
            prev_stage == next_stage
            and current_stage != prev_stage
        ):
            smoothed[i] = prev_stage

    return smoothed


# ============================================================
# 5. Metric helper
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred_raw,
):

    y_pred_smoothed = smooth_predictions(
        y_pred_raw
    )

    metrics = {
        "raw_accuracy":
            accuracy_score(
                y_true,
                y_pred_raw,
            ),

        "raw_balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred_raw,
            ),

        "raw_macro_f1":
            f1_score(
                y_true,
                y_pred_raw,
                average="macro",
            ),

        "smoothed_accuracy":
            accuracy_score(
                y_true,
                y_pred_smoothed,
            ),

        "smoothed_balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred_smoothed,
            ),

        "smoothed_macro_f1":
            f1_score(
                y_true,
                y_pred_smoothed,
                average="macro",
            ),
    }

    return (
        metrics,
        y_pred_smoothed,
    )


# ============================================================
# 6. Canonical training preprocessing
#
# Preserve the Stage 29/30 training pipeline.
# This isolates deployment-time normalization effects.
# ============================================================

train_normalized = (
    offline_recording_normalize(
        train_raw,
        feature_columns,
    )
)

X_train = train_normalized[
    feature_columns
]

y_train = train_normalized[
    "label"
]


# ============================================================
# 7. Train canonical 150-tree model
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=1,
)

print(
    "\nTraining canonical "
    "150-tree Random Forest..."
)

train_start = time.perf_counter()

model.fit(
    X_train,
    y_train,
)

train_end = time.perf_counter()

training_time_sec = (
    train_end
    - train_start
)

print(
    "Training time (sec):",
    round(
        training_time_sec,
        4,
    ),
)


# ============================================================
# 8. Build normalization candidates
# ============================================================

print("\nBuilding normalization candidates...")

training_means, training_stds = (
    training_global_statistics(
        train_raw,
        feature_columns,
    )
)

normalization_candidates = {}


# ---------- Offline baseline ----------

normalization_candidates[
    "offline_full_recording"
] = offline_recording_normalize(
    test_raw,
    feature_columns,
)


# ---------- Stage 31 causal expanding ----------

normalization_candidates[
    "causal_expanding"
] = causal_expanding_normalize(
    test_raw,
    feature_columns,
)


# ---------- Rolling windows ----------

normalization_candidates[
    "rolling_5min"
] = rolling_window_normalize(
    test_raw,
    feature_columns,
    window_epochs=10,
)

normalization_candidates[
    "rolling_10min"
] = rolling_window_normalize(
    test_raw,
    feature_columns,
    window_epochs=20,
)

normalization_candidates[
    "rolling_30min"
] = rolling_window_normalize(
    test_raw,
    feature_columns,
    window_epochs=60,
)


# ---------- Warm-up fixed baselines ----------

normalization_candidates[
    "warmup_5min"
] = warmup_baseline_normalize(
    test_raw,
    feature_columns,
    warmup_epochs=10,
)

normalization_candidates[
    "warmup_10min"
] = warmup_baseline_normalize(
    test_raw,
    feature_columns,
    warmup_epochs=20,
)

normalization_candidates[
    "warmup_30min"
] = warmup_baseline_normalize(
    test_raw,
    feature_columns,
    warmup_epochs=60,
)


# ---------- Training-global fixed stats ----------

normalization_candidates[
    "training_global_fixed"
] = fixed_statistics_normalize(
    test_raw,
    feature_columns,
    training_means,
    training_stds,
)


# ============================================================
# 9. Sanity checks
# ============================================================

for (
    method_name,
    normalized_df,
) in normalization_candidates.items():

    nan_count = int(
        normalized_df[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    inf_count = int(
        np.isinf(
            normalized_df[
                feature_columns
            ].to_numpy()
        ).sum()
    )

    print(
        f"{method_name:<24} "
        f"NaN={nan_count:<5} "
        f"Inf={inf_count:<5}"
    )

    if (
        nan_count != 0
        or inf_count != 0
    ):
        raise ValueError(
            f"Invalid values found in "
            f"{method_name}"
        )


# ============================================================
# 10. Evaluate all normalization strategies
# ============================================================

y_test = test_raw[
    "label"
]

results = []

prediction_table = test_raw[
    [
        "subject",
        "recording",
        "epoch",
        "start_sec",
        "label",
    ]
].copy()


print("\n" + "=" * 78)
print("NORMALIZATION BENCHMARK")
print("=" * 78)


for (
    method_name,
    normalized_df,
) in normalization_candidates.items():

    X_test = normalized_df[
        feature_columns
    ]

    inference_start = (
        time.perf_counter()
    )

    y_pred_raw = model.predict(
        X_test
    )

    inference_end = (
        time.perf_counter()
    )

    inference_sec = (
        inference_end
        - inference_start
    )

    (
        metrics,
        y_pred_smoothed,
    ) = evaluate_predictions(
        y_test,
        y_pred_raw,
    )

    prediction_table[
        f"{method_name}_raw"
    ] = y_pred_raw

    prediction_table[
        f"{method_name}_smoothed"
    ] = y_pred_smoothed

    result_row = {
        "method":
            method_name,

        "raw_accuracy":
            metrics[
                "raw_accuracy"
            ],

        "raw_balanced_accuracy":
            metrics[
                "raw_balanced_accuracy"
            ],

        "raw_macro_f1":
            metrics[
                "raw_macro_f1"
            ],

        "smoothed_accuracy":
            metrics[
                "smoothed_accuracy"
            ],

        "smoothed_balanced_accuracy":
            metrics[
                "smoothed_balanced_accuracy"
            ],

        "smoothed_macro_f1":
            metrics[
                "smoothed_macro_f1"
            ],

        "inference_sec":
            inference_sec,
    }

    results.append(
        result_row
    )

    print(
        f"\n{method_name}"
    )

    print(
        "  Raw Macro F1:",
        round(
            metrics[
                "raw_macro_f1"
            ],
            4,
        ),
    )

    print(
        "  Smoothed Macro F1:",
        round(
            metrics[
                "smoothed_macro_f1"
            ],
            4,
        ),
    )

    print(
        "  Smoothed Balanced Accuracy:",
        round(
            metrics[
                "smoothed_balanced_accuracy"
            ],
            4,
        ),
    )

    print(
        "  Inference time (sec):",
        round(
            inference_sec,
            6,
        ),
    )


# ============================================================
# 11. Build result dataframe
# ============================================================

results_df = pd.DataFrame(
    results
)

offline_row = results_df[
    results_df[
        "method"
    ] == "offline_full_recording"
].iloc[0]

offline_smoothed_macro_f1 = float(
    offline_row[
        "smoothed_macro_f1"
    ]
)

results_df[
    "smoothed_macro_f1_delta_vs_offline"
] = (
    results_df[
        "smoothed_macro_f1"
    ]
    - offline_smoothed_macro_f1
)


# ============================================================
# 12. Deployment eligibility
# ============================================================

deployable_methods = [
    "causal_expanding",
    "rolling_5min",
    "rolling_10min",
    "rolling_30min",
    "warmup_5min",
    "warmup_10min",
    "warmup_30min",
    "training_global_fixed",
]

results_df[
    "deployable"
] = results_df[
    "method"
].isin(
    deployable_methods
)


# ============================================================
# 13. Ranking
# ============================================================

deployable_results = (
    results_df[
        results_df[
            "deployable"
        ]
    ]
    .sort_values(
        by=[
            "smoothed_macro_f1",
            "smoothed_balanced_accuracy",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

deployable_results[
    "deployment_rank"
] = (
    np.arange(
        len(
            deployable_results
        )
    )
    + 1
)

rank_map = dict(
    zip(
        deployable_results[
            "method"
        ],
        deployable_results[
            "deployment_rank"
        ],
    )
)

results_df[
    "deployment_rank"
] = results_df[
    "method"
].map(
    rank_map
)


# ============================================================
# 14. Best deployable strategy
# ============================================================

best_row = deployable_results.iloc[
    0
]

best_method = str(
    best_row[
        "method"
    ]
)

best_smoothed_macro_f1 = float(
    best_row[
        "smoothed_macro_f1"
    ]
)

best_gap = float(
    best_row[
        "smoothed_macro_f1_delta_vs_offline"
    ]
)

print("\n" + "=" * 78)
print("DEPLOYMENT RANKING")
print("=" * 78)

display_columns = [
    "deployment_rank",
    "method",
    "smoothed_macro_f1",
    "smoothed_balanced_accuracy",
    "smoothed_macro_f1_delta_vs_offline",
]

print(
    deployable_results[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 15. Prediction-change analysis
# ============================================================

offline_predictions = (
    prediction_table[
        "offline_full_recording_smoothed"
    ]
)

prediction_change_rows = []

for method in deployable_methods:

    method_predictions = (
        prediction_table[
            f"{method}_smoothed"
        ]
    )

    changed = (
        method_predictions
        != offline_predictions
    )

    changed_count = int(
        changed.sum()
    )

    changed_percent = float(
        changed_count
        / len(changed)
        * 100
    )

    prediction_change_rows.append(
        {
            "method":
                method,

            "changed_epoch_count":
                changed_count,

            "changed_epoch_percent":
                changed_percent,
        }
    )

prediction_change_df = pd.DataFrame(
    prediction_change_rows
)

results_df = results_df.merge(
    prediction_change_df,
    on="method",
    how="left",
)


# ============================================================
# 16. Save benchmark CSV
# ============================================================

benchmark_path = (
    results_dir
    / "stage32_normalization_benchmark.csv"
)

results_df.to_csv(
    benchmark_path,
    index=False,
)


# ============================================================
# 17. Save epoch-level predictions
# ============================================================

prediction_path = (
    results_dir
    / "stage32_epoch_predictions.csv"
)

prediction_table.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# 18. Save training statistics
#
# Useful if training_global_fixed becomes competitive.
# These values could later be packaged with the model.
# ============================================================

training_stats_rows = []

for feature in feature_columns:

    training_stats_rows.append(
        {
            "feature":
                feature,

            "training_mean":
                training_means[
                    feature
                ],

            "training_std":
                training_stds[
                    feature
                ],
        }
    )

training_stats_df = pd.DataFrame(
    training_stats_rows
)

training_stats_path = (
    results_dir
    / "stage32_training_global_statistics.csv"
)

training_stats_df.to_csv(
    training_stats_path,
    index=False,
)


# ============================================================
# 19. Save compact winner summary
# ============================================================

winner_summary_df = pd.DataFrame([
    {
        "test_subject":
            test_subject,

        "tree_count":
            model.n_estimators,

        "feature_count":
            len(feature_columns),

        "training_time_sec":
            training_time_sec,

        "offline_smoothed_macro_f1":
            offline_smoothed_macro_f1,

        "best_deployable_method":
            best_method,

        "best_deployable_smoothed_macro_f1":
            best_smoothed_macro_f1,

        "best_gap_vs_offline":
            best_gap,

        "best_deployment_rank":
            1,
    }
])

winner_summary_path = (
    results_dir
    / "stage32_winner_summary.csv"
)

winner_summary_df.to_csv(
    winner_summary_path,
    index=False,
)


# ============================================================
# 20. Final interpretation
# ============================================================

print("\n" + "=" * 78)
print("STAGE 32 SUMMARY")
print("=" * 78)

print(
    "\nOffline baseline "
    "Smoothed Macro F1:",
    round(
        offline_smoothed_macro_f1,
        4,
    ),
)

print(
    "\nBest deployable method:"
)

print(
    best_method
)

print(
    "Best deployable "
    "Smoothed Macro F1:",
    round(
        best_smoothed_macro_f1,
        4,
    ),
)

print(
    "Gap vs offline:",
    round(
        best_gap,
        4,
    ),
)


# ============================================================
# 21. Simple decision guidance
# ============================================================

absolute_gap = abs(
    best_gap
)

print("\nDeployment interpretation:")

if absolute_gap <= 0.02:

    print(
        "EXCELLENT: best deployable normalization "
        "is within 0.02 Macro F1 of offline baseline."
    )

    print(
        "The model is close to a realistic "
        "real-time preprocessing pipeline."
    )

elif absolute_gap <= 0.05:

    print(
        "GOOD: deployment gap is moderate "
        "and likely manageable."
    )

    print(
        "Proceed with the best normalization strategy "
        "and validate it across more subjects."
    )

elif absolute_gap <= 0.10:

    print(
        "CAUTION: meaningful deployment gap remains."
    )

    print(
        "The strategy is usable for prototype work, "
        "but further normalization tuning is recommended."
    )

else:

    print(
        "WARNING: large deployment gap remains."
    )

    print(
        "Do not claim real-time equivalence to "
        "offline performance yet."
    )


# ============================================================
# 22. Output locations
# ============================================================

print("\nSaved benchmark to:")
print(benchmark_path)

print("\nSaved epoch predictions to:")
print(prediction_path)

print("\nSaved training statistics to:")
print(training_stats_path)

print("\nSaved winner summary to:")
print(winner_summary_path)

print("\n" + "=" * 78)
print("STAGE 32 COMPLETE")
print("=" * 78)
