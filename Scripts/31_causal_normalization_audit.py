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

print("=" * 70)
print("STAGE 31 - CAUSAL NORMALIZATION DEPLOYMENT AUDIT")
print("=" * 70)

print("\nTest subject:")
print(test_subject)

print("\nFeature count:")
print(len(feature_columns))

print("\nSubjects:")
print(subjects)


# ============================================================
# 2. Offline recording-relative normalization
#    Same normalization used in Stage 29 / 30
# ============================================================

def offline_recording_normalize(
    input_df,
    features,
):
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


# ============================================================
# 3. Causal expanding normalization
#
# Each epoch can only use the current and previous epochs
# from the same recording.
#
# This is closer to a real-time wearable scenario because
# future epochs are not available.
# ============================================================

def causal_expanding_normalize(
    input_df,
    features,
):
    output_df = input_df.copy()

    # Preserve original row order after grouped processing
    original_index = output_df.index

    for feature in features:

        causal_values = pd.Series(
            index=output_df.index,
            dtype=float
        )

        for recording, group in output_df.groupby(
            "recording",
            sort=False
        ):

            values = group[feature].astype(float)

            expanding_mean = (
                values
                .expanding(min_periods=1)
                .mean()
            )

            expanding_std = (
                values
                .expanding(min_periods=2)
                .std()
            )

            normalized = (
                values - expanding_mean
            ) / (
                expanding_std + 1e-8
            )

            # First epoch has no valid standard deviation.
            # A neutral standardized value of 0 is used.
            normalized = (
                normalized
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
            )

            causal_values.loc[group.index] = normalized

        output_df[feature] = causal_values.loc[
            original_index
        ].values

    return output_df


# ============================================================
# 4. Build train / test split
# ============================================================

train_raw = df_raw[
    df_raw["subject"] != test_subject
].copy()

test_raw = df_raw[
    df_raw["subject"] == test_subject
].copy()

print(
    "\nTraining epochs:",
    len(train_raw)
)

print(
    "Test epochs:",
    len(test_raw)
)


# ============================================================
# 5. Training preprocessing
#
# Keep Stage 29 training pipeline unchanged so Stage 31
# isolates the effect of deployment-time normalization.
# ============================================================

train_offline = offline_recording_normalize(
    train_raw,
    feature_columns
)

X_train = train_offline[
    feature_columns
]

y_train = train_offline[
    "label"
]


# ============================================================
# 6. Two test preprocessing modes
# ============================================================

test_offline = offline_recording_normalize(
    test_raw,
    feature_columns
)

test_causal = causal_expanding_normalize(
    test_raw,
    feature_columns
)

X_test_offline = test_offline[
    feature_columns
]

X_test_causal = test_causal[
    feature_columns
]

y_test = test_raw[
    "label"
]


# ============================================================
# 7. Safety checks
# ============================================================

offline_nan_count = int(
    X_test_offline
    .isna()
    .sum()
    .sum()
)

causal_nan_count = int(
    X_test_causal
    .isna()
    .sum()
    .sum()
)

print(
    "\nOffline test NaN count:",
    offline_nan_count
)

print(
    "Causal test NaN count:",
    causal_nan_count
)

if offline_nan_count != 0:
    raise ValueError(
        "Offline-normalized test data contains NaN."
    )

if causal_nan_count != 0:
    raise ValueError(
        "Causal-normalized test data contains NaN."
    )


# ============================================================
# 8. Train canonical lightweight model
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=1,
)

print("\nTraining canonical 150-tree model...")

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
# 9. Conservative A-B-A smoothing
# ============================================================

def smooth_predictions(predictions):

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
# 10. Evaluation helper
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred_raw,
):

    y_pred_smoothed = smooth_predictions(
        y_pred_raw
    )

    return {
        "raw_accuracy":
            accuracy_score(
                y_true,
                y_pred_raw
            ),

        "raw_balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred_raw
            ),

        "raw_macro_f1":
            f1_score(
                y_true,
                y_pred_raw,
                average="macro"
            ),

        "smoothed_accuracy":
            accuracy_score(
                y_true,
                y_pred_smoothed
            ),

        "smoothed_balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred_smoothed
            ),

        "smoothed_macro_f1":
            f1_score(
                y_true,
                y_pred_smoothed,
                average="macro"
            ),
    }


# ============================================================
# 11. Offline benchmark
# ============================================================

offline_start = time.perf_counter()

offline_pred = model.predict(
    X_test_offline
)

offline_end = time.perf_counter()

offline_inference_sec = (
    offline_end - offline_start
)

offline_metrics = evaluate_predictions(
    y_test,
    offline_pred
)


# ============================================================
# 12. Causal benchmark
# ============================================================

causal_start = time.perf_counter()

causal_pred = model.predict(
    X_test_causal
)

causal_end = time.perf_counter()

causal_inference_sec = (
    causal_end - causal_start
)

causal_metrics = evaluate_predictions(
    y_test,
    causal_pred
)


# ============================================================
# 13. Compare preprocessing values
# ============================================================

difference = (
    X_test_offline.to_numpy()
    - X_test_causal.to_numpy()
)

mean_abs_feature_difference = float(
    np.mean(
        np.abs(
            difference
        )
    )
)

max_abs_feature_difference = float(
    np.max(
        np.abs(
            difference
        )
    )
)


# ============================================================
# 14. Print results
# ============================================================

print("\n" + "=" * 70)
print("OFFLINE NORMALIZATION")
print("=" * 70)

print(
    "Raw Accuracy:",
    round(
        offline_metrics[
            "raw_accuracy"
        ],
        4
    )
)

print(
    "Raw Balanced Accuracy:",
    round(
        offline_metrics[
            "raw_balanced_accuracy"
        ],
        4
    )
)

print(
    "Raw Macro F1:",
    round(
        offline_metrics[
            "raw_macro_f1"
        ],
        4
    )
)

print(
    "Smoothed Accuracy:",
    round(
        offline_metrics[
            "smoothed_accuracy"
        ],
        4
    )
)

print(
    "Smoothed Balanced Accuracy:",
    round(
        offline_metrics[
            "smoothed_balanced_accuracy"
        ],
        4
    )
)

print(
    "Smoothed Macro F1:",
    round(
        offline_metrics[
            "smoothed_macro_f1"
        ],
        4
    )
)


print("\n" + "=" * 70)
print("CAUSAL NORMALIZATION")
print("=" * 70)

print(
    "Raw Accuracy:",
    round(
        causal_metrics[
            "raw_accuracy"
        ],
        4
    )
)

print(
    "Raw Balanced Accuracy:",
    round(
        causal_metrics[
            "raw_balanced_accuracy"
        ],
        4
    )
)

print(
    "Raw Macro F1:",
    round(
        causal_metrics[
            "raw_macro_f1"
        ],
        4
    )
)

print(
    "Smoothed Accuracy:",
    round(
        causal_metrics[
            "smoothed_accuracy"
        ],
        4
    )
)

print(
    "Smoothed Balanced Accuracy:",
    round(
        causal_metrics[
            "smoothed_balanced_accuracy"
        ],
        4
    )
)

print(
    "Smoothed Macro F1:",
    round(
        causal_metrics[
            "smoothed_macro_f1"
        ],
        4
    )
)


# ============================================================
# 15. Deployment gap
# ============================================================

raw_macro_f1_delta = (
    causal_metrics[
        "raw_macro_f1"
    ]
    - offline_metrics[
        "raw_macro_f1"
    ]
)

smoothed_macro_f1_delta = (
    causal_metrics[
        "smoothed_macro_f1"
    ]
    - offline_metrics[
        "smoothed_macro_f1"
    ]
)


print("\n" + "=" * 70)
print("DEPLOYMENT GAP")
print("=" * 70)

print(
    "Raw Macro F1 delta "
    "(causal - offline):",
    round(
        raw_macro_f1_delta,
        4
    )
)

print(
    "Smoothed Macro F1 delta "
    "(causal - offline):",
    round(
        smoothed_macro_f1_delta,
        4
    )
)

print(
    "Mean absolute normalized-feature difference:",
    round(
        mean_abs_feature_difference,
        6
    )
)

print(
    "Max absolute normalized-feature difference:",
    round(
        max_abs_feature_difference,
        6
    )
)


# ============================================================
# 16. Save summary CSV
# ============================================================

summary_df = pd.DataFrame([
    {
        "subject":
            test_subject,

        "feature_count":
            len(feature_columns),

        "tree_count":
            model.n_estimators,

        "training_time_sec":
            training_time_sec,

        "offline_inference_sec":
            offline_inference_sec,

        "causal_inference_sec":
            causal_inference_sec,

        "offline_raw_accuracy":
            offline_metrics[
                "raw_accuracy"
            ],

        "causal_raw_accuracy":
            causal_metrics[
                "raw_accuracy"
            ],

        "offline_raw_balanced_accuracy":
            offline_metrics[
                "raw_balanced_accuracy"
            ],

        "causal_raw_balanced_accuracy":
            causal_metrics[
                "raw_balanced_accuracy"
            ],

        "offline_raw_macro_f1":
            offline_metrics[
                "raw_macro_f1"
            ],

        "causal_raw_macro_f1":
            causal_metrics[
                "raw_macro_f1"
            ],

        "offline_smoothed_accuracy":
            offline_metrics[
                "smoothed_accuracy"
            ],

        "causal_smoothed_accuracy":
            causal_metrics[
                "smoothed_accuracy"
            ],

        "offline_smoothed_balanced_accuracy":
            offline_metrics[
                "smoothed_balanced_accuracy"
            ],

        "causal_smoothed_balanced_accuracy":
            causal_metrics[
                "smoothed_balanced_accuracy"
            ],

        "offline_smoothed_macro_f1":
            offline_metrics[
                "smoothed_macro_f1"
            ],

        "causal_smoothed_macro_f1":
            causal_metrics[
                "smoothed_macro_f1"
            ],

        "raw_macro_f1_delta":
            raw_macro_f1_delta,

        "smoothed_macro_f1_delta":
            smoothed_macro_f1_delta,

        "mean_abs_feature_difference":
            mean_abs_feature_difference,

        "max_abs_feature_difference":
            max_abs_feature_difference,
    }
])

summary_path = (
    results_dir
    / "stage31_causal_normalization_audit.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 17. Save epoch-level comparison
# ============================================================

epoch_comparison_df = test_raw[
    [
        "subject",
        "recording",
        "epoch",
        "start_sec",
        "label",
    ]
].copy()

epoch_comparison_df[
    "offline_prediction"
] = offline_pred

epoch_comparison_df[
    "causal_prediction"
] = causal_pred

epoch_comparison_df[
    "prediction_changed"
] = (
    offline_pred
    != causal_pred
)

epoch_output_path = (
    results_dir
    / "stage31_epoch_comparison.csv"
)

epoch_comparison_df.to_csv(
    epoch_output_path,
    index=False
)


# ============================================================
# 18. Final summary
# ============================================================

changed_prediction_count = int(
    epoch_comparison_df[
        "prediction_changed"
    ].sum()
)

changed_prediction_percent = (
    changed_prediction_count
    / len(epoch_comparison_df)
    * 100
)

print("\nPrediction changes:")
print(
    changed_prediction_count,
    "/",
    len(epoch_comparison_df),
    f"({changed_prediction_percent:.2f}%)"
)

print("\nSaved summary to:")
print(summary_path)

print("\nSaved epoch comparison to:")
print(epoch_output_path)

print("\n" + "=" * 70)
print("STAGE 31 COMPLETE")
print("=" * 70)

print(
    "\nInterpretation:"
)

print(
    "Stage 29/30 uses full-recording normalization, "
    "which has access to the complete recording."
)

print(
    "Stage 31 compares it with causal expanding normalization, "
    "where each epoch only uses information available up to "
    "that point in time."
)

print(
    "If causal Macro F1 remains close to offline Macro F1, "
    "the current model is much closer to real-time deployment."
)

print(
    "If performance drops substantially, normalization must be "
    "redesigned before claiming real-time wearable inference."
)
