from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
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

results_dir = (
    PROJECT_ROOT
    / "results"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Stage 37
# Hierarchical Classifier Benchmark
#
# Compare:
#
# A. baseline_5class
#    Direct Wake / N1 / N2 / N3 / REM classification
#
# B. n1_first
#    Stage 1: N1 vs non-N1
#    Stage 2: non-N1 -> Wake / N2 / N3 / REM
#
# C. transitional_branch
#    Stage 1: stable vs transitional
#       stable       = N2 / N3
#       transitional = Wake / N1 / REM
#
#    Stage 2a:
#       stable -> N2 / N3
#
#    Stage 2b:
#       transitional -> Wake / N1 / REM
#
# All pipelines use:
#   - LOSO
#   - training-global fixed normalization per fold
#   - 150-tree Random Forest components
#   - same features
#   - recording-aware A-B-A smoothing
#
# Goal:
# Determine whether architecture, rather than missing features,
# is the main reason N1 collapses in multiclass classification.
# ============================================================


print("=" * 88)
print("STAGE 37 - HIERARCHICAL CLASSIFIER BENCHMARK")
print("=" * 88)


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
    c
    for c in df_raw.columns
    if c not in metadata_columns
]

subjects = sorted(
    df_raw[
        "subject"
    ].unique()
)

stage_order = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]

stage_order = [
    stage
    for stage in stage_order
    if stage in set(
        df_raw[
            "label"
        ].unique()
    )
]

print("\nSubjects:")
print(subjects)

print("\nFeature count:")
print(len(feature_columns))

print("\nTotal epochs:")
print(len(df_raw))

print("\nStages:")
print(stage_order)


# ============================================================
# 2. Training statistics
# ============================================================

def calculate_training_statistics(
    train_df,
    features,
):

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
# 3. Fixed normalization
# ============================================================

def normalize_with_fixed_statistics(
    input_df,
    features,
    means,
    stds,
):

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
# 4. Recording-aware A-B-A smoothing
# ============================================================

def smooth_predictions_by_recording(
    prediction_df,
):

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
                smoothed[
                    i
                ] = prev_stage

        output.loc[
            indices,
            "smoothed_prediction",
        ] = smoothed

    return output


# ============================================================
# 5. Metric helpers
# ============================================================

def overall_metrics(
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


def per_class_metrics(
    y_true,
    y_pred,
    labels,
):

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    result = {}

    for (
        stage,
        p,
        r,
        f,
        s,
    ) in zip(
        labels,
        precision,
        recall,
        f1,
        support,
    ):

        result[
            stage
        ] = {
            "precision":
                float(p),

            "recall":
                float(r),

            "f1":
                float(f),

            "support":
                int(s),
        }

    return result


# ============================================================
# 6. RF factory
# ============================================================

def make_rf():
    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


# ============================================================
# 7. Architecture A - baseline direct 5-class
# ============================================================

def predict_baseline_5class(
    X_train,
    y_train,
    X_test,
):

    model = make_rf()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    return predictions


# ============================================================
# 8. Architecture B - N1-first hierarchy
# ============================================================

def predict_n1_first(
    X_train,
    y_train,
    X_test,
):

    # --------------------------------------------------------
    # Stage 1:
    # N1 vs non-N1
    # --------------------------------------------------------

    y_binary = (
        y_train == "N1"
    ).astype(int)

    model_n1 = make_rf()

    model_n1.fit(
        X_train,
        y_binary,
    )

    pred_n1_binary = model_n1.predict(
        X_test
    )

    final_predictions = np.empty(
        len(X_test),
        dtype=object,
    )

    n1_mask = (
        pred_n1_binary == 1
    )

    final_predictions[
        n1_mask
    ] = "N1"

    # --------------------------------------------------------
    # Stage 2:
    # non-N1 -> Wake / N2 / N3 / REM
    # --------------------------------------------------------

    non_n1_train_mask = (
        y_train != "N1"
    )

    X_train_non_n1 = X_train[
        non_n1_train_mask
    ]

    y_train_non_n1 = y_train[
        non_n1_train_mask
    ]

    model_non_n1 = make_rf()

    model_non_n1.fit(
        X_train_non_n1,
        y_train_non_n1,
    )

    non_n1_test_mask = (
        ~n1_mask
    )

    if int(
        non_n1_test_mask.sum()
    ) > 0:

        final_predictions[
            non_n1_test_mask
        ] = model_non_n1.predict(
            X_test[
                non_n1_test_mask
            ]
        )

    return final_predictions


# ============================================================
# 9. Architecture C - transitional branch
# ============================================================

def predict_transitional_branch(
    X_train,
    y_train,
    X_test,
):

    stable_classes = {
        "N2",
        "N3",
    }

    transitional_classes = {
        "Wake",
        "N1",
        "REM",
    }

    # --------------------------------------------------------
    # Stage 1:
    # stable vs transitional
    # --------------------------------------------------------

    y_branch = y_train.map(
        lambda stage:
            "stable"
            if stage in stable_classes
            else "transitional"
    )

    model_branch = make_rf()

    model_branch.fit(
        X_train,
        y_branch,
    )

    pred_branch = model_branch.predict(
        X_test
    )

    final_predictions = np.empty(
        len(X_test),
        dtype=object,
    )

    stable_test_mask = (
        pred_branch == "stable"
    )

    transitional_test_mask = (
        pred_branch == "transitional"
    )

    # --------------------------------------------------------
    # Stage 2a:
    # stable -> N2 / N3
    # --------------------------------------------------------

    stable_train_mask = y_train.isin(
        stable_classes
    )

    model_stable = make_rf()

    model_stable.fit(
        X_train[
            stable_train_mask
        ],
        y_train[
            stable_train_mask
        ],
    )

    if int(
        stable_test_mask.sum()
    ) > 0:

        final_predictions[
            stable_test_mask
        ] = model_stable.predict(
            X_test[
                stable_test_mask
            ]
        )

    # --------------------------------------------------------
    # Stage 2b:
    # transitional -> Wake / N1 / REM
    # --------------------------------------------------------

    transitional_train_mask = y_train.isin(
        transitional_classes
    )

    model_transitional = make_rf()

    model_transitional.fit(
        X_train[
            transitional_train_mask
        ],
        y_train[
            transitional_train_mask
        ],
    )

    if int(
        transitional_test_mask.sum()
    ) > 0:

        final_predictions[
            transitional_test_mask
        ] = model_transitional.predict(
            X_test[
                transitional_test_mask
            ]
        )

    return final_predictions


# ============================================================
# 10. Architecture registry
# ============================================================

architectures = {
    "baseline_5class":
        predict_baseline_5class,

    "n1_first":
        predict_n1_first,

    "transitional_branch":
        predict_transitional_branch,
}


# ============================================================
# 11. LOSO benchmark
# ============================================================

all_fold_rows = []
all_epoch_prediction_frames = []


print("\n" + "=" * 88)
print("STARTING LOSO ARCHITECTURE BENCHMARK")
print("=" * 88)


for architecture_name, predictor in architectures.items():

    print("\n" + "#" * 88)

    print(
        "Architecture:",
        architecture_name
    )

    print("#" * 88)

    for fold_idx, test_subject in enumerate(
        subjects,
        start=1,
    ):

        print(
            f"\nFold "
            f"{fold_idx}/{len(subjects)} "
            f"- Test subject: "
            f"{test_subject}"
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Training-only normalization statistics
        # ----------------------------------------------------

        means, stds = (
            calculate_training_statistics(
                train_raw,
                feature_columns,
            )
        )

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

        # ----------------------------------------------------
        # Predict architecture
        # ----------------------------------------------------

        start = time.perf_counter()

        y_pred_raw = predictor(
            X_train,
            y_train,
            X_test,
        )

        end = time.perf_counter()

        architecture_time_sec = (
            end - start
        )

        # ----------------------------------------------------
        # Smoothing
        # ----------------------------------------------------

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

        y_pred_smoothed = (
            prediction_df[
                "smoothed_prediction"
            ]
            .to_numpy()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        raw_metrics = overall_metrics(
            y_test,
            y_pred_raw,
        )

        smoothed_metrics = overall_metrics(
            y_test,
            y_pred_smoothed,
        )

        per_class = per_class_metrics(
            y_test,
            y_pred_smoothed,
            stage_order,
        )

        fold_row = {
            "architecture":
                architecture_name,

            "test_subject":
                test_subject,

            "test_epoch_count":
                len(
                    X_test
                ),

            "architecture_time_sec":
                architecture_time_sec,

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
        }

        for stage in stage_order:

            fold_row[
                f"{stage}_precision"
            ] = per_class[
                stage
            ][
                "precision"
            ]

            fold_row[
                f"{stage}_recall"
            ] = per_class[
                stage
            ][
                "recall"
            ]

            fold_row[
                f"{stage}_f1"
            ] = per_class[
                stage
            ][
                "f1"
            ]

        all_fold_rows.append(
            fold_row
        )

        prediction_df[
            "architecture"
        ] = architecture_name

        prediction_df[
            "test_subject_fold"
        ] = test_subject

        all_epoch_prediction_frames.append(
            prediction_df
        )

        print(
            "  Smoothed Macro F1:",
            round(
                smoothed_metrics[
                    "macro_f1"
                ],
                4,
            )
        )

        print(
            "  N1 F1:",
            round(
                per_class[
                    "N1"
                ][
                    "f1"
                ],
                4,
            )
        )


# ============================================================
# 12. Dataframes
# ============================================================

fold_results_df = pd.DataFrame(
    all_fold_rows
)

epoch_predictions_df = pd.concat(
    all_epoch_prediction_frames,
    ignore_index=True,
)


# ============================================================
# 13. Architecture aggregate results
# ============================================================

aggregate_rows = []


for architecture_name in architectures.keys():

    subset = fold_results_df[
        fold_results_df[
            "architecture"
        ] == architecture_name
    ]

    pooled_df = epoch_predictions_df[
        epoch_predictions_df[
            "architecture"
        ] == architecture_name
    ]

    pooled_metrics = overall_metrics(
        pooled_df[
            "label"
        ],
        pooled_df[
            "smoothed_prediction"
        ],
    )

    pooled_per_class = per_class_metrics(
        pooled_df[
            "label"
        ],
        pooled_df[
            "smoothed_prediction"
        ],
        stage_order,
    )

    row = {
        "architecture":
            architecture_name,

        "mean_subject_macro_f1":
            float(
                subset[
                    "smoothed_macro_f1"
                ].mean()
            ),

        "std_subject_macro_f1":
            float(
                subset[
                    "smoothed_macro_f1"
                ].std(
                    ddof=0
                )
            ),

        "min_subject_macro_f1":
            float(
                subset[
                    "smoothed_macro_f1"
                ].min()
            ),

        "max_subject_macro_f1":
            float(
                subset[
                    "smoothed_macro_f1"
                ].max()
            ),

        "mean_balanced_accuracy":
            float(
                subset[
                    "smoothed_balanced_accuracy"
                ].mean()
            ),

        "pooled_accuracy":
            pooled_metrics[
                "accuracy"
            ],

        "pooled_balanced_accuracy":
            pooled_metrics[
                "balanced_accuracy"
            ],

        "pooled_macro_f1":
            pooled_metrics[
                "macro_f1"
            ],

        "mean_architecture_time_sec":
            float(
                subset[
                    "architecture_time_sec"
                ].mean()
            ),
    }

    for stage in stage_order:

        row[
            f"{stage}_precision"
        ] = pooled_per_class[
            stage
        ][
            "precision"
        ]

        row[
            f"{stage}_recall"
        ] = pooled_per_class[
            stage
        ][
            "recall"
        ]

        row[
            f"{stage}_f1"
        ] = pooled_per_class[
            stage
        ][
            "f1"
        ]

    aggregate_rows.append(
        row
    )


aggregate_df = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# 14. Baseline deltas
# ============================================================

baseline_row = aggregate_df[
    aggregate_df[
        "architecture"
    ] == "baseline_5class"
].iloc[0]


aggregate_df[
    "delta_mean_subject_macro_f1"
] = (
    aggregate_df[
        "mean_subject_macro_f1"
    ]
    - float(
        baseline_row[
            "mean_subject_macro_f1"
        ]
    )
)

aggregate_df[
    "delta_pooled_macro_f1"
] = (
    aggregate_df[
        "pooled_macro_f1"
    ]
    - float(
        baseline_row[
            "pooled_macro_f1"
        ]
    )
)

aggregate_df[
    "delta_N1_f1"
] = (
    aggregate_df[
        "N1_f1"
    ]
    - float(
        baseline_row[
            "N1_f1"
        ]
    )
)


# ============================================================
# 15. Ranking
# ============================================================

ranking_df = (
    aggregate_df
    .sort_values(
        by=[
            "mean_subject_macro_f1",
            "pooled_macro_f1",
            "min_subject_macro_f1",
        ],
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
        len(
            ranking_df
        )
    )
    + 1
)


# ============================================================
# 16. Print ranking
# ============================================================

print("\n" + "=" * 88)
print("STAGE 37 ARCHITECTURE RANKING")
print("=" * 88)

display_columns = [
    "architecture",
    "mean_subject_macro_f1",
    "pooled_macro_f1",
    "min_subject_macro_f1",
    "N1_f1",
    "Wake_f1",
    "N2_f1",
    "N3_f1",
    "REM_f1",
    "delta_mean_subject_macro_f1",
    "delta_N1_f1",
    "mean_architecture_time_sec",
]

print(
    ranking_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 17. Best architecture subject comparison
# ============================================================

best_architecture = str(
    ranking_df.iloc[
        0
    ][
        "architecture"
    ]
)

baseline_subject_df = (
    fold_results_df[
        fold_results_df[
            "architecture"
        ] == "baseline_5class"
    ][
        [
            "test_subject",
            "smoothed_macro_f1",
            "N1_f1",
        ]
    ]
    .rename(
        columns={
            "smoothed_macro_f1":
                "baseline_macro_f1",

            "N1_f1":
                "baseline_N1_f1",
        }
    )
)

best_subject_df = (
    fold_results_df[
        fold_results_df[
            "architecture"
        ] == best_architecture
    ][
        [
            "test_subject",
            "smoothed_macro_f1",
            "N1_f1",
        ]
    ]
    .rename(
        columns={
            "smoothed_macro_f1":
                "best_macro_f1",

            "N1_f1":
                "best_N1_f1",
        }
    )
)

subject_comparison_df = (
    baseline_subject_df
    .merge(
        best_subject_df,
        on="test_subject",
        how="inner",
    )
)

subject_comparison_df[
    "macro_f1_delta"
] = (
    subject_comparison_df[
        "best_macro_f1"
    ]
    - subject_comparison_df[
        "baseline_macro_f1"
    ]
)

subject_comparison_df[
    "N1_f1_delta"
] = (
    subject_comparison_df[
        "best_N1_f1"
    ]
    - subject_comparison_df[
        "baseline_N1_f1"
    ]
)


print("\n" + "=" * 88)
print("BEST ARCHITECTURE SUBJECT COMPARISON")
print("=" * 88)

print(
    subject_comparison_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 18. Final summary
# ============================================================

best_row = ranking_df.iloc[
    0
]

best_mean_macro_f1 = float(
    best_row[
        "mean_subject_macro_f1"
    ]
)

best_n1_f1 = float(
    best_row[
        "N1_f1"
    ]
)

best_macro_delta = float(
    best_row[
        "delta_mean_subject_macro_f1"
    ]
)

best_n1_delta = float(
    best_row[
        "delta_N1_f1"
    ]
)


print("\n" + "=" * 88)
print("STAGE 37 SUMMARY")
print("=" * 88)

print(
    "\nBest architecture:",
    best_architecture
)

print(
    "Best mean LOSO Macro F1:",
    round(
        best_mean_macro_f1,
        4,
    )
)

print(
    "Best pooled N1 F1:",
    round(
        best_n1_f1,
        4,
    )
)

print(
    "Macro F1 delta vs baseline:",
    round(
        best_macro_delta,
        4,
    )
)

print(
    "N1 F1 delta vs baseline:",
    round(
        best_n1_delta,
        4,
    )
)


print("\nInterpretation:")

if best_architecture == "baseline_5class":

    print(
        "The direct 5-class model remains best."
    )

    print(
        "Hierarchical decomposition did not improve "
        "overall cross-subject performance."
    )

elif (
    best_macro_delta > 0.01
    and best_n1_delta > 0.03
):

    print(
        "PROMISING:"
    )

    print(
        "Hierarchical classification improves both "
        "overall LOSO Macro F1 and N1 recognition."
    )

    print(
        "This architecture is a strong candidate "
        "for the next canonical pipeline."
    )

elif best_macro_delta > 0:

    print(
        "MARGINAL IMPROVEMENT:"
    )

    print(
        "Hierarchy provides some benefit, "
        "but the gain is limited."
    )

else:

    print(
        "NO OVERALL BENEFIT:"
    )

    print(
        "The hierarchy may help N1 locally but "
        "does not improve the full multiclass pipeline."
    )


# ============================================================
# 19. Save outputs
# ============================================================

fold_results_path = (
    results_dir
    / "stage37_fold_results.csv"
)

aggregate_path = (
    results_dir
    / "stage37_architecture_benchmark.csv"
)

ranking_path = (
    results_dir
    / "stage37_architecture_ranking.csv"
)

subject_comparison_path = (
    results_dir
    / "stage37_best_subject_comparison.csv"
)

epoch_predictions_path = (
    results_dir
    / "stage37_epoch_predictions.csv"
)

summary_path = (
    results_dir
    / "stage37_summary.csv"
)

fold_results_df.to_csv(
    fold_results_path,
    index=False,
)

aggregate_df.to_csv(
    aggregate_path,
    index=False,
)

ranking_df.to_csv(
    ranking_path,
    index=False,
)

subject_comparison_df.to_csv(
    subject_comparison_path,
    index=False,
)

epoch_predictions_df.to_csv(
    epoch_predictions_path,
    index=False,
)

pd.DataFrame([
    {
        "best_architecture":
            best_architecture,

        "best_mean_loso_macro_f1":
            best_mean_macro_f1,

        "best_pooled_n1_f1":
            best_n1_f1,

        "macro_f1_delta_vs_baseline":
            best_macro_delta,

        "n1_f1_delta_vs_baseline":
            best_n1_delta,
    }
]).to_csv(
    summary_path,
    index=False,
)


print("\nSaved fold results to:")
print(fold_results_path)

print("\nSaved architecture benchmark to:")
print(aggregate_path)

print("\nSaved ranking to:")
print(ranking_path)

print("\nSaved best subject comparison to:")
print(subject_comparison_path)

print("\nSaved epoch predictions to:")
print(epoch_predictions_path)

print("\nSaved summary to:")
print(summary_path)

print("\n" + "=" * 88)
print("STAGE 37 COMPLETE")
print("=" * 88)
