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
# Stage 33
# LOSO validation of deployment-consistent
# training-global fixed normalization
#
# Important methodological rule:
#
# For EACH LOSO fold:
#   1. Hold out one subject completely.
#   2. Calculate mean/std ONLY from training subjects.
#   3. Apply the SAME training mean/std to:
#         - training data
#         - held-out test data
#   4. Train the canonical 150-tree RF.
#   5. Evaluate raw + A-B-A smoothed predictions.
#
# This prevents:
#   - test-subject leakage
#   - future-recording leakage
#   - train/test preprocessing mismatch
# ============================================================


print("=" * 80)
print("STAGE 33 - LOSO DEPLOYMENT NORMALIZATION VALIDATION")
print("=" * 80)


# ============================================================
# 1. Load dataset
# ============================================================

df_raw = pd.read_csv(
    data_path
)

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

print("\nSubjects:")
print(subjects)

print("\nSubject count:")
print(len(subjects))

print("\nFeature count:")
print(len(feature_columns))

print("\nTotal epochs:")
print(len(df_raw))


# ============================================================
# 2. Safety checks
# ============================================================

raw_nan_count = int(
    df_raw[
        feature_columns
    ]
    .isna()
    .sum()
    .sum()
)

raw_inf_count = int(
    np.isinf(
        df_raw[
            feature_columns
        ].to_numpy()
    ).sum()
)

print(
    "\nRaw NaN count:",
    raw_nan_count
)

print(
    "Raw Inf count:",
    raw_inf_count
)

if (
    raw_nan_count != 0
    or raw_inf_count != 0
):
    raise ValueError(
        "Raw feature table contains NaN or Inf."
    )


# ============================================================
# 3. Training-global statistics
# ============================================================

def calculate_training_statistics(
    train_df,
    features,
):
    """
    Calculate mean/std using TRAINING DATA ONLY.

    These statistics are then used for both training
    and held-out test data in the current LOSO fold.
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


# ============================================================
# 4. Fixed-statistics normalization
# ============================================================

def normalize_with_fixed_statistics(
    input_df,
    features,
    means,
    stds,
):
    """
    Apply fixed training-set mean/std.

    No statistics are calculated from the test subject.
    """

    output_df = input_df.copy()

    for feature in features:

        output_df[feature] = (
            output_df[
                feature
            ].astype(float)
            - means[feature]
        ) / (
            stds[feature]
            + 1e-8
        )

    output_df[
        features
    ] = (
        output_df[
            features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return output_df


# ============================================================
# 5. Conservative A-B-A smoothing
# ============================================================

def smooth_predictions_by_recording(
    prediction_df,
):
    """
    Apply A-B-A smoothing separately inside each recording.

    This is safer than smoothing across recording boundaries.
    """

    output = prediction_df.copy()

    output[
        "smoothed_prediction"
    ] = output[
        "raw_prediction"
    ].copy()

    for recording, group in output.groupby(
        "recording",
        sort=False,
    ):

        indices = group.index.to_list()

        predictions = (
            group[
                "raw_prediction"
            ]
            .to_numpy()
            .copy()
        )

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

        output.loc[
            indices,
            "smoothed_prediction",
        ] = smoothed

    return output


# ============================================================
# 6. Metric helper
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
            ),
    }


# ============================================================
# 7. LOSO loop
# ============================================================

fold_results = []

all_epoch_predictions = []

all_training_statistics = []


print("\n" + "=" * 80)
print("STARTING LOSO VALIDATION")
print("=" * 80)


for fold_idx, test_subject in enumerate(
    subjects,
    start=1,
):

    print("\n" + "-" * 80)

    print(
        f"Fold {fold_idx}/{len(subjects)}"
    )

    print(
        "Held-out subject:",
        test_subject
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_raw = df_raw[
        df_raw[
            "subject"
        ] != test_subject
    ].copy()

    test_raw = df_raw[
        df_raw[
            "subject"
        ] == test_subject
    ].copy()

    print(
        "Training epochs:",
        len(train_raw)
    )

    print(
        "Test epochs:",
        len(test_raw)
    )

    # --------------------------------------------------------
    # Training-only statistics
    # --------------------------------------------------------

    means, stds = (
        calculate_training_statistics(
            train_raw,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # Store fold-specific statistics
    # --------------------------------------------------------

    for feature in feature_columns:

        all_training_statistics.append(
            {
                "test_subject":
                    test_subject,

                "feature":
                    feature,

                "training_mean":
                    means[
                        feature
                    ],

                "training_std":
                    stds[
                        feature
                    ],
            }
        )

    # --------------------------------------------------------
    # Apply SAME normalization to train and test
    # --------------------------------------------------------

    train_normalized = (
        normalize_with_fixed_statistics(
            train_raw,
            feature_columns,
            means,
            stds,
        )
    )

    test_normalized = (
        normalize_with_fixed_statistics(
            test_raw,
            feature_columns,
            means,
            stds,
        )
    )

    X_train = train_normalized[
        feature_columns
    ]

    y_train = train_normalized[
        "label"
    ]

    X_test = test_normalized[
        feature_columns
    ]

    y_test = test_normalized[
        "label"
    ]

    # --------------------------------------------------------
    # Check normalization
    # --------------------------------------------------------

    train_nan_count = int(
        X_train
        .isna()
        .sum()
        .sum()
    )

    test_nan_count = int(
        X_test
        .isna()
        .sum()
        .sum()
    )

    if (
        train_nan_count != 0
        or test_nan_count != 0
    ):
        raise ValueError(
            f"NaN found in fold "
            f"{test_subject}"
        )

    # --------------------------------------------------------
    # Canonical 150-tree Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
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

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    inference_start = (
        time.perf_counter()
    )

    y_pred_raw = model.predict(
        X_test
    )

    inference_end = (
        time.perf_counter()
    )

    inference_time_sec = (
        inference_end
        - inference_start
    )

    # --------------------------------------------------------
    # Recording-aware smoothing
    # --------------------------------------------------------

    prediction_df = test_raw[
        [
            "subject",
            "recording",
            "epoch",
            "start_sec",
            "label",
        ]
    ].copy()

    prediction_df[
        "raw_prediction"
    ] = y_pred_raw

    prediction_df = (
        smooth_predictions_by_recording(
            prediction_df
        )
    )

    y_pred_smoothed = prediction_df[
        "smoothed_prediction"
    ].to_numpy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    raw_metrics = calculate_metrics(
        y_test,
        y_pred_raw,
    )

    smoothed_metrics = (
        calculate_metrics(
            y_test,
            y_pred_smoothed,
        )
    )

    smoothing_delta = (
        smoothed_metrics[
            "macro_f1"
        ]
        - raw_metrics[
            "macro_f1"
        ]
    )

    # --------------------------------------------------------
    # Prediction changes caused by smoothing
    # --------------------------------------------------------

    changed_by_smoothing = int(
        (
            prediction_df[
                "raw_prediction"
            ]
            != prediction_df[
                "smoothed_prediction"
            ]
        ).sum()
    )

    changed_by_smoothing_percent = (
        changed_by_smoothing
        / len(prediction_df)
        * 100
    )

    # --------------------------------------------------------
    # Fold result
    # --------------------------------------------------------

    fold_result = {
        "test_subject":
            test_subject,

        "training_epoch_count":
            len(X_train),

        "test_epoch_count":
            len(X_test),

        "feature_count":
            len(feature_columns),

        "tree_count":
            model.n_estimators,

        "training_time_sec":
            training_time_sec,

        "inference_time_sec":
            inference_time_sec,

        "raw_accuracy":
            raw_metrics[
                "accuracy"
            ],

        "raw_balanced_accuracy":
            raw_metrics[
                "balanced_accuracy"
            ],

        "raw_macro_f1":
            raw_metrics[
                "macro_f1"
            ],

        "smoothed_accuracy":
            smoothed_metrics[
                "accuracy"
            ],

        "smoothed_balanced_accuracy":
            smoothed_metrics[
                "balanced_accuracy"
            ],

        "smoothed_macro_f1":
            smoothed_metrics[
                "macro_f1"
            ],

        "smoothing_macro_f1_delta":
            smoothing_delta,

        "smoothing_changed_epochs":
            changed_by_smoothing,

        "smoothing_changed_percent":
            changed_by_smoothing_percent,
    }

    fold_results.append(
        fold_result
    )

    prediction_df[
        "test_subject_fold"
    ] = test_subject

    all_epoch_predictions.append(
        prediction_df
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "Raw Macro F1:",
        round(
            raw_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Smoothed Macro F1:",
        round(
            smoothed_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Smoothed Balanced Accuracy:",
        round(
            smoothed_metrics[
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print(
        "Smoothing delta:",
        round(
            smoothing_delta,
            4,
        ),
    )

    print(
        "Training time (sec):",
        round(
            training_time_sec,
            4,
        ),
    )

    print(
        "Inference time (sec):",
        round(
            inference_time_sec,
            6,
        ),
    )


# ============================================================
# 8. Build LOSO results table
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_df = (
    fold_results_df
    .sort_values(
        by="test_subject"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. Aggregate statistics
# ============================================================

metric_columns = [
    "raw_accuracy",
    "raw_balanced_accuracy",
    "raw_macro_f1",
    "smoothed_accuracy",
    "smoothed_balanced_accuracy",
    "smoothed_macro_f1",
]

aggregate_rows = []

for metric in metric_columns:

    values = fold_results_df[
        metric
    ].to_numpy()

    aggregate_rows.append(
        {
            "metric":
                metric,

            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "std":
                float(
                    np.std(
                        values
                    )
                ),

            "median":
                float(
                    np.median(
                        values
                    )
                ),

            "min":
                float(
                    np.min(
                        values
                    )
                ),

            "max":
                float(
                    np.max(
                        values
                    )
                ),
        }
    )

aggregate_df = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# 10. Macro F1 subject ranking
# ============================================================

ranking_df = (
    fold_results_df[
        [
            "test_subject",
            "smoothed_macro_f1",
            "smoothed_balanced_accuracy",
            "smoothed_accuracy",
        ]
    ]
    .sort_values(
        by="smoothed_macro_f1",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

ranking_df[
    "rank"
] = (
    np.arange(
        len(ranking_df)
    )
    + 1
)

ranking_df = ranking_df[
    [
        "rank",
        "test_subject",
        "smoothed_macro_f1",
        "smoothed_balanced_accuracy",
        "smoothed_accuracy",
    ]
]


# ============================================================
# 11. Pooled epoch-level metrics
#
# These metrics combine predictions from every held-out fold.
# They complement, but do not replace, mean per-subject metrics.
# ============================================================

all_epoch_predictions_df = pd.concat(
    all_epoch_predictions,
    ignore_index=True,
)

pooled_raw_metrics = calculate_metrics(
    all_epoch_predictions_df[
        "label"
    ],
    all_epoch_predictions_df[
        "raw_prediction"
    ],
)

pooled_smoothed_metrics = (
    calculate_metrics(
        all_epoch_predictions_df[
            "label"
        ],
        all_epoch_predictions_df[
            "smoothed_prediction"
        ],
    )
)


# ============================================================
# 12. Summary values
# ============================================================

mean_smoothed_macro_f1 = float(
    fold_results_df[
        "smoothed_macro_f1"
    ].mean()
)

std_smoothed_macro_f1 = float(
    fold_results_df[
        "smoothed_macro_f1"
    ].std(
        ddof=0
    )
)

median_smoothed_macro_f1 = float(
    fold_results_df[
        "smoothed_macro_f1"
    ].median()
)

min_smoothed_macro_f1 = float(
    fold_results_df[
        "smoothed_macro_f1"
    ].min()
)

max_smoothed_macro_f1 = float(
    fold_results_df[
        "smoothed_macro_f1"
    ].max()
)

mean_smoothed_balanced_accuracy = float(
    fold_results_df[
        "smoothed_balanced_accuracy"
    ].mean()
)

best_subject = str(
    ranking_df.iloc[
        0
    ][
        "test_subject"
    ]
)

best_subject_f1 = float(
    ranking_df.iloc[
        0
    ][
        "smoothed_macro_f1"
    ]
)

worst_subject = str(
    ranking_df.iloc[
        -1
    ][
        "test_subject"
    ]
)

worst_subject_f1 = float(
    ranking_df.iloc[
        -1
    ][
        "smoothed_macro_f1"
    ]
)


# ============================================================
# 13. Print ranking
# ============================================================

print("\n" + "=" * 80)
print("LOSO SUBJECT RANKING")
print("=" * 80)

print(
    ranking_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 14. Print aggregate results
# ============================================================

print("\n" + "=" * 80)
print("STAGE 33 AGGREGATE RESULTS")
print("=" * 80)

print(
    "\nMean Smoothed Macro F1:",
    round(
        mean_smoothed_macro_f1,
        4,
    ),
)

print(
    "Std Smoothed Macro F1:",
    round(
        std_smoothed_macro_f1,
        4,
    ),
)

print(
    "Median Smoothed Macro F1:",
    round(
        median_smoothed_macro_f1,
        4,
    ),
)

print(
    "Min Smoothed Macro F1:",
    round(
        min_smoothed_macro_f1,
        4,
    ),
)

print(
    "Max Smoothed Macro F1:",
    round(
        max_smoothed_macro_f1,
        4,
    ),
)

print(
    "\nMean Smoothed Balanced Accuracy:",
    round(
        mean_smoothed_balanced_accuracy,
        4,
    ),
)

print(
    "\nPooled Smoothed Accuracy:",
    round(
        pooled_smoothed_metrics[
            "accuracy"
        ],
        4,
    ),
)

print(
    "Pooled Smoothed Balanced Accuracy:",
    round(
        pooled_smoothed_metrics[
            "balanced_accuracy"
        ],
        4,
    ),
)

print(
    "Pooled Smoothed Macro F1:",
    round(
        pooled_smoothed_metrics[
            "macro_f1"
        ],
        4,
    ),
)

print(
    "\nBest subject:",
    best_subject,
    "| Macro F1:",
    round(
        best_subject_f1,
        4,
    ),
)

print(
    "Worst subject:",
    worst_subject,
    "| Macro F1:",
    round(
        worst_subject_f1,
        4,
    ),
)


# ============================================================
# 15. Stability assessment
# ============================================================

print("\n" + "=" * 80)
print("DEPLOYMENT STABILITY ASSESSMENT")
print("=" * 80)

if (
    mean_smoothed_macro_f1 >= 0.75
    and min_smoothed_macro_f1 >= 0.60
):

    print(
        "STRONG:"
    )

    print(
        "Deployment-consistent fixed normalization "
        "generalizes well across held-out subjects."
    )

elif (
    mean_smoothed_macro_f1 >= 0.70
    and min_smoothed_macro_f1 >= 0.50
):

    print(
        "PROMISING:"
    )

    print(
        "Cross-subject deployment performance is "
        "reasonable for prototype validation, "
        "but subject variability remains."
    )

elif (
    mean_smoothed_macro_f1 >= 0.60
):

    print(
        "CAUTION:"
    )

    print(
        "The pipeline shows useful signal, "
        "but cross-subject robustness requires "
        "additional improvement."
    )

else:

    print(
        "WARNING:"
    )

    print(
        "Cross-subject generalization is currently "
        "too weak for a strong deployment claim."
    )


# ============================================================
# 16. Save outputs
# ============================================================

fold_results_path = (
    results_dir
    / "stage33_loso_fold_results.csv"
)

fold_results_df.to_csv(
    fold_results_path,
    index=False,
)


aggregate_path = (
    results_dir
    / "stage33_loso_aggregate.csv"
)

aggregate_df.to_csv(
    aggregate_path,
    index=False,
)


ranking_path = (
    results_dir
    / "stage33_subject_ranking.csv"
)

ranking_df.to_csv(
    ranking_path,
    index=False,
)


epoch_predictions_path = (
    results_dir
    / "stage33_loso_epoch_predictions.csv"
)

all_epoch_predictions_df.to_csv(
    epoch_predictions_path,
    index=False,
)


statistics_df = pd.DataFrame(
    all_training_statistics
)

statistics_path = (
    results_dir
    / "stage33_fold_training_statistics.csv"
)

statistics_df.to_csv(
    statistics_path,
    index=False,
)


summary_df = pd.DataFrame([
    {
        "subject_count":
            len(subjects),

        "feature_count":
            len(feature_columns),

        "tree_count":
            150,

        "normalization":
            "training_global_fixed_per_fold",

        "mean_smoothed_macro_f1":
            mean_smoothed_macro_f1,

        "std_smoothed_macro_f1":
            std_smoothed_macro_f1,

        "median_smoothed_macro_f1":
            median_smoothed_macro_f1,

        "min_smoothed_macro_f1":
            min_smoothed_macro_f1,

        "max_smoothed_macro_f1":
            max_smoothed_macro_f1,

        "mean_smoothed_balanced_accuracy":
            mean_smoothed_balanced_accuracy,

        "pooled_smoothed_accuracy":
            pooled_smoothed_metrics[
                "accuracy"
            ],

        "pooled_smoothed_balanced_accuracy":
            pooled_smoothed_metrics[
                "balanced_accuracy"
            ],

        "pooled_smoothed_macro_f1":
            pooled_smoothed_metrics[
                "macro_f1"
            ],

        "best_subject":
            best_subject,

        "best_subject_macro_f1":
            best_subject_f1,

        "worst_subject":
            worst_subject,

        "worst_subject_macro_f1":
            worst_subject_f1,
    }
])

summary_path = (
    results_dir
    / "stage33_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# 17. Output locations
# ============================================================

print("\nSaved fold results to:")
print(fold_results_path)

print("\nSaved aggregate metrics to:")
print(aggregate_path)

print("\nSaved subject ranking to:")
print(ranking_path)

print("\nSaved epoch predictions to:")
print(epoch_predictions_path)

print("\nSaved fold training statistics to:")
print(statistics_path)

print("\nSaved summary to:")
print(summary_path)

print("\n" + "=" * 80)
print("STAGE 33 COMPLETE")
print("=" * 80)
