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
# Stage 35
# N1 Cost-Sensitive Learning Benchmark
#
# Goal:
# Keep the deployment-consistent Stage 33 pipeline fixed,
# and only adjust N1 class weighting.
#
# Fixed components:
#   - LOSO validation
#   - training-global normalization per fold
#   - 150-tree Random Forest
#   - same feature set
#   - recording-aware A-B-A smoothing
#
# Variable:
#   - N1 weight multiplier
# ============================================================


print("=" * 84)
print("STAGE 35 - N1 COST-SENSITIVE LEARNING BENCHMARK")
print("=" * 84)


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
# 2. Candidate N1 multipliers
# ============================================================

n1_multipliers = [
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
]


# ============================================================
# 3. Training-global statistics
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
# 4. Fixed normalization
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
# 5. Recording-aware A-B-A smoothing
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
# 6. Metric helpers
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
# 7. Build baseline class weights per fold
#
# We use sklearn-style balanced weights calculated manually,
# then multiply ONLY N1 by the candidate factor.
#
# balanced weight:
#   n_samples / (n_classes * class_count)
# ============================================================

def build_class_weight(
    y_train,
    n1_multiplier,
):

    classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    total = len(
        y_train
    )

    n_classes = len(
        classes
    )

    class_weight = {}

    for cls, count in zip(
        classes,
        counts,
    ):

        weight = (
            total
            / (
                n_classes
                * count
            )
        )

        if cls == "N1":
            weight = (
                weight
                * n1_multiplier
            )

        class_weight[
            cls
        ] = float(
            weight
        )

    return class_weight


# ============================================================
# 8. Benchmark loop
# ============================================================

all_results = []
all_subject_results = []
all_epoch_predictions = []


print("\n" + "=" * 84)
print("STARTING N1 WEIGHT BENCHMARK")
print("=" * 84)


for n1_multiplier in n1_multipliers:

    print("\n" + "#" * 84)

    print(
        "N1 multiplier:",
        n1_multiplier
    )

    print("#" * 84)

    multiplier_subject_rows = []

    multiplier_epoch_predictions = []

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

        # ----------------------------------------------------
        # Normalize train and test using SAME stats
        # ----------------------------------------------------

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
        # Build cost-sensitive class weights
        # ----------------------------------------------------

        class_weight = (
            build_class_weight(
                y_train,
                n1_multiplier,
            )
        )

        # ----------------------------------------------------
        # Train RF
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight=class_weight,
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

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

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
            ].to_numpy()
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

        smoothed_per_class = (
            per_class_metrics(
                y_test,
                y_pred_smoothed,
                stage_order,
            )
        )

        # ----------------------------------------------------
        # Subject result
        # ----------------------------------------------------

        subject_row = {
            "n1_multiplier":
                n1_multiplier,

            "test_subject":
                test_subject,

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
        }

        for stage in stage_order:

            subject_row[
                f"{stage}_precision"
            ] = smoothed_per_class[
                stage
            ][
                "precision"
            ]

            subject_row[
                f"{stage}_recall"
            ] = smoothed_per_class[
                stage
            ][
                "recall"
            ]

            subject_row[
                f"{stage}_f1"
            ] = smoothed_per_class[
                stage
            ][
                "f1"
            ]

            subject_row[
                f"{stage}_support"
            ] = smoothed_per_class[
                stage
            ][
                "support"
            ]

        multiplier_subject_rows.append(
            subject_row
        )

        all_subject_results.append(
            subject_row
        )

        prediction_df[
            "n1_multiplier"
        ] = n1_multiplier

        prediction_df[
            "test_subject_fold"
        ] = test_subject

        multiplier_epoch_predictions.append(
            prediction_df
        )

        all_epoch_predictions.append(
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
                smoothed_per_class[
                    "N1"
                ][
                    "f1"
                ],
                4,
            )
        )

    # ========================================================
    # Aggregate current multiplier across subjects
    # ========================================================

    multiplier_subject_df = pd.DataFrame(
        multiplier_subject_rows
    )

    pooled_prediction_df = pd.concat(
        multiplier_epoch_predictions,
        ignore_index=True,
    )

    pooled_overall = overall_metrics(
        pooled_prediction_df[
            "label"
        ],
        pooled_prediction_df[
            "smoothed_prediction"
        ],
    )

    pooled_per_class = per_class_metrics(
        pooled_prediction_df[
            "label"
        ],
        pooled_prediction_df[
            "smoothed_prediction"
        ],
        stage_order,
    )

    result_row = {
        "n1_multiplier":
            n1_multiplier,

        "mean_subject_macro_f1":
            float(
                multiplier_subject_df[
                    "smoothed_macro_f1"
                ].mean()
            ),

        "std_subject_macro_f1":
            float(
                multiplier_subject_df[
                    "smoothed_macro_f1"
                ].std(
                    ddof=0
                )
            ),

        "min_subject_macro_f1":
            float(
                multiplier_subject_df[
                    "smoothed_macro_f1"
                ].min()
            ),

        "max_subject_macro_f1":
            float(
                multiplier_subject_df[
                    "smoothed_macro_f1"
                ].max()
            ),

        "pooled_accuracy":
            pooled_overall[
                "accuracy"
            ],

        "pooled_balanced_accuracy":
            pooled_overall[
                "balanced_accuracy"
            ],

        "pooled_macro_f1":
            pooled_overall[
                "macro_f1"
            ],
    }

    for stage in stage_order:

        result_row[
            f"{stage}_precision"
        ] = pooled_per_class[
            stage
        ][
            "precision"
        ]

        result_row[
            f"{stage}_recall"
        ] = pooled_per_class[
            stage
        ][
            "recall"
        ]

        result_row[
            f"{stage}_f1"
        ] = pooled_per_class[
            stage
        ][
            "f1"
        ]

    all_results.append(
        result_row
    )

    print("\nAggregate for multiplier:")
    print(
        "  Mean subject Macro F1:",
        round(
            result_row[
                "mean_subject_macro_f1"
            ],
            4,
        )
    )

    print(
        "  Pooled Macro F1:",
        round(
            result_row[
                "pooled_macro_f1"
            ],
            4,
        )
    )

    print(
        "  Pooled N1 F1:",
        round(
            result_row[
                "N1_f1"
            ],
            4,
        )
    )

    print(
        "  Pooled REM F1:",
        round(
            result_row[
                "REM_f1"
            ],
            4,
        )
    )


# ============================================================
# 9. Build benchmark dataframe
# ============================================================

results_df = pd.DataFrame(
    all_results
)

subject_results_df = pd.DataFrame(
    all_subject_results
)

epoch_predictions_df = pd.concat(
    all_epoch_predictions,
    ignore_index=True,
)


# ============================================================
# 10. Calculate deltas vs baseline multiplier 1.0
# ============================================================

baseline_row = results_df[
    results_df[
        "n1_multiplier"
    ] == 1.0
].iloc[0]


results_df[
    "delta_mean_subject_macro_f1"
] = (
    results_df[
        "mean_subject_macro_f1"
    ]
    - float(
        baseline_row[
            "mean_subject_macro_f1"
        ]
    )
)

results_df[
    "delta_pooled_macro_f1"
] = (
    results_df[
        "pooled_macro_f1"
    ]
    - float(
        baseline_row[
            "pooled_macro_f1"
        ]
    )
)

results_df[
    "delta_N1_f1"
] = (
    results_df[
        "N1_f1"
    ]
    - float(
        baseline_row[
            "N1_f1"
        ]
    )
)

for stage in stage_order:

    if stage == "N1":
        continue

    results_df[
        f"delta_{stage}_f1"
    ] = (
        results_df[
            f"{stage}_f1"
        ]
        - float(
            baseline_row[
                f"{stage}_f1"
            ]
        )
    )


# ============================================================
# 11. Ranking
#
# Primary target:
#   maximize mean LOSO subject Macro F1
#
# Tie-breakers:
#   pooled Macro F1
#   minimum subject Macro F1
# ============================================================

ranking_df = (
    results_df
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
# 12. Best candidate
# ============================================================

best_row = ranking_df.iloc[
    0
]

best_multiplier = float(
    best_row[
        "n1_multiplier"
    ]
)

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

best_delta_macro_f1 = float(
    best_row[
        "delta_mean_subject_macro_f1"
    ]
)

best_delta_n1_f1 = float(
    best_row[
        "delta_N1_f1"
    ]
)


# ============================================================
# 13. Print compact ranking
# ============================================================

print("\n" + "=" * 84)
print("STAGE 35 RANKING")
print("=" * 84)

display_columns = [
    "n1_multiplier",
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
# 14. Best candidate subject-level comparison
# ============================================================

baseline_subject_df = (
    subject_results_df[
        subject_results_df[
            "n1_multiplier"
        ] == 1.0
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
    subject_results_df[
        subject_results_df[
            "n1_multiplier"
        ] == best_multiplier
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


print("\n" + "=" * 84)
print("BEST MULTIPLIER SUBJECT COMPARISON")
print("=" * 84)

print(
    subject_comparison_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 15. Decision logic
# ============================================================

print("\n" + "=" * 84)
print("STAGE 35 SUMMARY")
print("=" * 84)

print(
    "\nBest N1 multiplier:",
    best_multiplier
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
    "Macro F1 improvement vs baseline:",
    round(
        best_delta_macro_f1,
        4,
    )
)

print(
    "N1 F1 improvement vs baseline:",
    round(
        best_delta_n1_f1,
        4,
    )
)


print("\nInterpretation:")

if (
    best_multiplier == 1.0
):

    print(
        "No N1 upweighting candidate improved "
        "overall LOSO Macro F1."
    )

    print(
        "Class weighting alone is not sufficient."
    )

elif (
    best_delta_macro_f1 > 0.01
    and best_delta_n1_f1 > 0.03
):

    print(
        "PROMISING:"
    )

    print(
        "N1 upweighting improves both N1 recognition "
        "and overall cross-subject Macro F1."
    )

    print(
        "This multiplier is a strong candidate "
        "for the next canonical model."
    )

elif (
    best_delta_macro_f1 > 0
):

    print(
        "MARGINAL:"
    )

    print(
        "N1 weighting provides a small overall benefit."
    )

    print(
        "Further feature or temporal improvement "
        "is still likely needed."
    )

else:

    print(
        "NO OVERALL BENEFIT:"
    )

    print(
        "N1 weighting may improve N1 locally "
        "but harms other classes or subjects."
    )


# ============================================================
# 16. Save outputs
# ============================================================

benchmark_path = (
    results_dir
    / "stage35_n1_weight_benchmark.csv"
)

results_df.to_csv(
    benchmark_path,
    index=False,
)


ranking_path = (
    results_dir
    / "stage35_n1_weight_ranking.csv"
)

ranking_df.to_csv(
    ranking_path,
    index=False,
)


subject_results_path = (
    results_dir
    / "stage35_subject_results.csv"
)

subject_results_df.to_csv(
    subject_results_path,
    index=False,
)


subject_comparison_path = (
    results_dir
    / "stage35_best_subject_comparison.csv"
)

subject_comparison_df.to_csv(
    subject_comparison_path,
    index=False,
)


epoch_predictions_path = (
    results_dir
    / "stage35_epoch_predictions.csv"
)

epoch_predictions_df.to_csv(
    epoch_predictions_path,
    index=False,
)


summary_df = pd.DataFrame([
    {
        "best_n1_multiplier":
            best_multiplier,

        "best_mean_loso_macro_f1":
            best_mean_macro_f1,

        "best_pooled_n1_f1":
            best_n1_f1,

        "macro_f1_delta_vs_baseline":
            best_delta_macro_f1,

        "n1_f1_delta_vs_baseline":
            best_delta_n1_f1,
    }
])

summary_path = (
    results_dir
    / "stage35_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print("\nSaved benchmark to:")
print(benchmark_path)

print("\nSaved ranking to:")
print(ranking_path)

print("\nSaved subject results to:")
print(subject_results_path)

print("\nSaved best-candidate subject comparison to:")
print(subject_comparison_path)

print("\nSaved epoch predictions to:")
print(epoch_predictions_path)

print("\nSaved summary to:")
print(summary_path)

print("\n" + "=" * 84)
print("STAGE 35 COMPLETE")
print("=" * 84)
