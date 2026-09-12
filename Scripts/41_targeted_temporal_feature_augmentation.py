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
# Stage 41
# Targeted Temporal Feature Augmentation Benchmark
#
# Goal:
#   Test whether causal feature dynamics improve the Stage 40
#   N1-preserving Viterbi pipeline.
#
# IMPORTANT:
#   No future epochs are used.
#
# Derived features are generated only when the dataset contains
# matching triplets:
#
#   current_X
#   prev1_X
#   prev2_X
#
# For each matched base feature X:
#
#   delta1_X
#       current_X - prev1_X
#
#   delta2_X
#       prev1_X - prev2_X
#
#   acceleration_X
#       delta1_X - delta2_X
#
#   ratio_current_prev1_X
#       current_X / (abs(prev1_X) + epsilon)
#
# These encode short-term transitions that may help distinguish
# transitional N1 from Wake, REM and N2.
#
# Compare:
#
#   A. stage40_base_features
#   B. stage41_augmented_features
#
# Both pipelines use:
#   - training-global normalization
#   - 150-tree RF
#   - nested LOSO
#   - nested selection of transition_weight
#   - nested selection of N1 emission boost
#   - N1-preserving Viterbi decoding
# ============================================================


print("=" * 100)
print("STAGE 41 - TARGETED TEMPORAL FEATURE AUGMENTATION BENCHMARK")
print("=" * 100)


# ============================================================
# 1. Load dataset
# ============================================================

df_original = pd.read_csv(
    data_path
)

metadata_columns = [
    "subject",
    "recording",
    "epoch",
    "start_sec",
    "label",
]

base_feature_columns = [
    c
    for c in df_original.columns
    if c not in metadata_columns
]

subjects = sorted(
    df_original[
        "subject"
    ].unique()
)

preferred_stage_order = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]

stage_order = [
    stage
    for stage in preferred_stage_order
    if stage in set(
        df_original[
            "label"
        ].unique()
    )
]

stage_to_idx = {
    stage: idx
    for idx, stage in enumerate(
        stage_order
    )
}

n1_index = stage_to_idx[
    "N1"
]


print("\nSubjects:")
print(subjects)

print("\nBase feature count:")
print(len(base_feature_columns))

print("\nTotal epochs:")
print(len(df_original))


# ============================================================
# 2. Find current/prev1/prev2 triplets
# ============================================================

current_features = [
    feature
    for feature in base_feature_columns
    if feature.startswith(
        "current_"
    )
]

matched_bases = []

for current_feature in current_features:

    suffix = current_feature[
        len(
            "current_"
        ):
    ]

    prev1_feature = (
        "prev1_"
        + suffix
    )

    prev2_feature = (
        "prev2_"
        + suffix
    )

    if (
        prev1_feature
        in base_feature_columns
        and prev2_feature
        in base_feature_columns
    ):

        matched_bases.append(
            suffix
        )


print("\nMatched temporal feature triplets:")
print(len(matched_bases))

print(matched_bases)


if len(
    matched_bases
) == 0:

    raise ValueError(
        "No current_/prev1_/prev2_ feature triplets found."
    )


# ============================================================
# 3. Build augmented dataset
# ============================================================

def add_temporal_dynamic_features(
    input_df,
    matched_suffixes,
):

    output_df = input_df.copy()

    epsilon = 1e-8

    derived_feature_names = []

    for suffix in matched_suffixes:

        current_col = (
            "current_"
            + suffix
        )

        prev1_col = (
            "prev1_"
            + suffix
        )

        prev2_col = (
            "prev2_"
            + suffix
        )

        delta1_col = (
            "dyn_delta1_"
            + suffix
        )

        delta2_col = (
            "dyn_delta2_"
            + suffix
        )

        acceleration_col = (
            "dyn_acceleration_"
            + suffix
        )

        ratio_col = (
            "dyn_ratio_current_prev1_"
            + suffix
        )

        output_df[
            delta1_col
        ] = (
            output_df[
                current_col
            ].astype(float)
            - output_df[
                prev1_col
            ].astype(float)
        )

        output_df[
            delta2_col
        ] = (
            output_df[
                prev1_col
            ].astype(float)
            - output_df[
                prev2_col
            ].astype(float)
        )

        output_df[
            acceleration_col
        ] = (
            output_df[
                delta1_col
            ]
            - output_df[
                delta2_col
            ]
        )

        output_df[
            ratio_col
        ] = (
            output_df[
                current_col
            ].astype(float)
            / (
                np.abs(
                    output_df[
                        prev1_col
                    ].astype(float)
                )
                + epsilon
            )
        )

        derived_feature_names.extend(
            [
                delta1_col,
                delta2_col,
                acceleration_col,
                ratio_col,
            ]
        )

    output_df[
        derived_feature_names
    ] = (
        output_df[
            derived_feature_names
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    return (
        output_df,
        derived_feature_names,
    )


(
    df_augmented,
    derived_feature_columns,
) = add_temporal_dynamic_features(
    df_original,
    matched_bases,
)


augmented_feature_columns = (
    base_feature_columns
    + derived_feature_columns
)


print("\nDerived feature count:")
print(len(derived_feature_columns))

print("\nAugmented total feature count:")
print(len(augmented_feature_columns))


# ============================================================
# 4. Candidate nested Viterbi parameters
# ============================================================

transition_weight_candidates = [
    0.25,
    0.50,
    0.75,
    1.00,
]

n1_emission_boost_candidates = [
    1.00,
    1.10,
    1.20,
    1.35,
    1.50,
    1.75,
]


# ============================================================
# 5. Normalization
# ============================================================

def calculate_training_statistics(
    train_df,
    features,
):

    means = {}
    stds = {}

    for feature in features:

        means[
            feature
        ] = float(
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
            np.isnan(
                std_value
            )
            or std_value < 1e-8
        ):
            std_value = 1.0

        stds[
            feature
        ] = std_value

    return means, stds


def normalize_with_fixed_statistics(
    input_df,
    features,
    means,
    stds,
):

    output_df = input_df.copy()

    for feature in features:

        output_df[
            feature
        ] = (
            output_df[
                feature
            ].astype(float)
            - means[
                feature
            ]
        ) / (
            stds[
                feature
            ]
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
        .fillna(
            0.0
        )
    )

    return output_df


# ============================================================
# 6. RF model
# ============================================================

def make_model():

    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


# ============================================================
# 7. Metrics
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

    return {
        label: {
            "precision":
                float(p),

            "recall":
                float(r),

            "f1":
                float(f),

            "support":
                int(s),
        }
        for (
            label,
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
        )
    }


# ============================================================
# 8. Temporal priors
# ============================================================

def estimate_temporal_priors(
    train_df,
    stages,
    laplace=1.0,
):

    stage_index = {
        stage: idx
        for idx, stage in enumerate(
            stages
        )
    }

    n_states = len(
        stages
    )

    start_counts = np.full(
        n_states,
        laplace,
        dtype=float,
    )

    transition_counts = np.full(
        (
            n_states,
            n_states,
        ),
        laplace,
        dtype=float,
    )

    for recording, group in train_df.groupby(
        "recording",
        sort=False,
    ):

        ordered = group.sort_values(
            by=[
                "epoch",
                "start_sec",
            ],
            kind="stable",
        )

        labels = ordered[
            "label"
        ].to_numpy()

        if len(
            labels
        ) == 0:

            continue

        first_stage = labels[
            0
        ]

        if first_stage in stage_index:

            start_counts[
                stage_index[
                    first_stage
                ]
            ] += 1.0

        for i in range(
            len(
                labels
            ) - 1
        ):

            current_stage = labels[
                i
            ]

            next_stage = labels[
                i + 1
            ]

            if (
                current_stage in stage_index
                and next_stage in stage_index
            ):

                transition_counts[
                    stage_index[
                        current_stage
                    ],
                    stage_index[
                        next_stage
                    ],
                ] += 1.0

    start_probabilities = (
        start_counts
        / start_counts.sum()
    )

    transition_probabilities = (
        transition_counts
        / transition_counts.sum(
            axis=1,
            keepdims=True,
        )
    )

    return (
        start_probabilities,
        transition_probabilities,
    )


# ============================================================
# 9. Align probabilities
# ============================================================

def align_model_probabilities(
    model,
    probabilities,
    stages,
):

    aligned = np.full(
        (
            len(
                probabilities
            ),
            len(
                stages
            ),
        ),
        1e-12,
        dtype=float,
    )

    class_to_idx = {
        label: idx
        for idx, label in enumerate(
            model.classes_
        )
    }

    for stage_idx, stage in enumerate(
        stages
    ):

        if stage in class_to_idx:

            aligned[
                :,
                stage_idx
            ] = probabilities[
                :,
                class_to_idx[
                    stage
                ],
            ]

    aligned = np.clip(
        aligned,
        1e-12,
        1.0,
    )

    aligned = (
        aligned
        / aligned.sum(
            axis=1,
            keepdims=True,
        )
    )

    return aligned


# ============================================================
# 10. N1-preserving Viterbi
# ============================================================

def viterbi_decode_sequence(
    emission_probabilities,
    start_probabilities,
    transition_probabilities,
    transition_weight,
    n1_emission_boost,
):

    n_epochs = emission_probabilities.shape[
        0
    ]

    n_states = emission_probabilities.shape[
        1
    ]

    log_emission = np.log(
        np.clip(
            emission_probabilities,
            1e-12,
            1.0,
        )
    )

    log_emission[
        :,
        n1_index
    ] += np.log(
        n1_emission_boost
    )

    log_start = np.log(
        np.clip(
            start_probabilities,
            1e-12,
            1.0,
        )
    )

    log_transition = np.log(
        np.clip(
            transition_probabilities,
            1e-12,
            1.0,
        )
    )

    dp = np.full(
        (
            n_epochs,
            n_states,
        ),
        -np.inf,
        dtype=float,
    )

    backpointer = np.zeros(
        (
            n_epochs,
            n_states,
        ),
        dtype=int,
    )

    dp[
        0,
        :
    ] = (
        log_start
        + log_emission[
            0,
            :
        ]
    )

    for t in range(
        1,
        n_epochs,
    ):

        for current_state in range(
            n_states
        ):

            transition_scores = (
                dp[
                    t - 1,
                    :
                ]
                + transition_weight
                * log_transition[
                    :,
                    current_state
                ]
            )

            best_previous_state = int(
                np.argmax(
                    transition_scores
                )
            )

            dp[
                t,
                current_state
            ] = (
                transition_scores[
                    best_previous_state
                ]
                + log_emission[
                    t,
                    current_state
                ]
            )

            backpointer[
                t,
                current_state
            ] = (
                best_previous_state
            )

    path = np.zeros(
        n_epochs,
        dtype=int,
    )

    path[
        -1
    ] = int(
        np.argmax(
            dp[
                -1,
                :
            ]
        )
    )

    for t in range(
        n_epochs - 2,
        -1,
        -1,
    ):

        path[
            t
        ] = backpointer[
            t + 1,
            path[
                t + 1
            ],
        ]

    return path


def viterbi_decode_by_recording(
    metadata_df,
    aligned_probabilities,
    start_probabilities,
    transition_probabilities,
    transition_weight,
    n1_emission_boost,
    stages,
):

    predictions = np.empty(
        len(
            metadata_df
        ),
        dtype=object,
    )

    working = metadata_df.copy()

    working[
        "_row_position"
    ] = np.arange(
        len(
            working
        )
    )

    for recording, group in working.groupby(
        "recording",
        sort=False,
    ):

        ordered = group.sort_values(
            by=[
                "epoch",
                "start_sec",
            ],
            kind="stable",
        )

        positions = ordered[
            "_row_position"
        ].to_numpy(
            dtype=int
        )

        recording_probabilities = (
            aligned_probabilities[
                positions
            ]
        )

        path = viterbi_decode_sequence(
            recording_probabilities,
            start_probabilities,
            transition_probabilities,
            transition_weight,
            n1_emission_boost,
        )

        labels = np.array(
            [
                stages[
                    idx
                ]
                for idx in path
            ],
            dtype=object,
        )

        predictions[
            positions
        ] = labels

    return predictions


# ============================================================
# 11. Inner LOSO parameter selection
# ============================================================

def select_parameters_inner_loso(
    outer_train_df,
    outer_train_subjects,
    feature_columns,
):

    parameter_records = {}

    for weight in transition_weight_candidates:

        for boost in n1_emission_boost_candidates:

            parameter_records[
                (
                    weight,
                    boost,
                )
            ] = []

    for inner_test_subject in outer_train_subjects:

        inner_train_raw = outer_train_df[
            outer_train_df[
                "subject"
            ] != inner_test_subject
        ].copy()

        inner_test_raw = outer_train_df[
            outer_train_df[
                "subject"
            ] == inner_test_subject
        ].copy()

        means, stds = (
            calculate_training_statistics(
                inner_train_raw,
                feature_columns,
            )
        )

        inner_train = (
            normalize_with_fixed_statistics(
                inner_train_raw,
                feature_columns,
                means,
                stds,
            )
        )

        inner_test = (
            normalize_with_fixed_statistics(
                inner_test_raw,
                feature_columns,
                means,
                stds,
            )
        )

        X_train = inner_train[
            feature_columns
        ]

        y_train = inner_train[
            "label"
        ]

        X_test = inner_test[
            feature_columns
        ]

        y_test = inner_test[
            "label"
        ]

        model = make_model()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        aligned_probabilities = (
            align_model_probabilities(
                model,
                probabilities,
                stage_order,
            )
        )

        (
            start_probabilities,
            transition_probabilities,
        ) = estimate_temporal_priors(
            inner_train_raw,
            stage_order,
            laplace=1.0,
        )

        metadata_df = inner_test_raw[
            [
                "subject",
                "recording",
                "epoch",
                "start_sec",
                "label",
            ]
        ].copy()

        for weight in transition_weight_candidates:

            for boost in n1_emission_boost_candidates:

                predictions = (
                    viterbi_decode_by_recording(
                        metadata_df,
                        aligned_probabilities,
                        start_probabilities,
                        transition_probabilities,
                        weight,
                        boost,
                        stage_order,
                    )
                )

                overall = overall_metrics(
                    y_test,
                    predictions,
                )

                per_class = per_class_metrics(
                    y_test,
                    predictions,
                    stage_order,
                )

                minimum_class_f1 = min(
                    per_class[
                        stage
                    ][
                        "f1"
                    ]
                    for stage in stage_order
                )

                parameter_records[
                    (
                        weight,
                        boost,
                    )
                ].append(
                    {
                        "macro_f1":
                            overall[
                                "macro_f1"
                            ],

                        "balanced_accuracy":
                            overall[
                                "balanced_accuracy"
                            ],

                        "n1_f1":
                            per_class[
                                "N1"
                            ][
                                "f1"
                            ],

                        "min_class_f1":
                            minimum_class_f1,
                    }
                )

    rows = []

    for (
        weight,
        boost,
    ), records in parameter_records.items():

        temp_df = pd.DataFrame(
            records
        )

        rows.append(
            {
                "transition_weight":
                    weight,

                "n1_emission_boost":
                    boost,

                "mean_macro_f1":
                    float(
                        temp_df[
                            "macro_f1"
                        ].mean()
                    ),

                "mean_n1_f1":
                    float(
                        temp_df[
                            "n1_f1"
                        ].mean()
                    ),

                "mean_min_class_f1":
                    float(
                        temp_df[
                            "min_class_f1"
                        ].mean()
                    ),

                "mean_balanced_accuracy":
                    float(
                        temp_df[
                            "balanced_accuracy"
                        ].mean()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "mean_macro_f1",
                "mean_n1_f1",
                "mean_min_class_f1",
                "mean_balanced_accuracy",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    best_weight = float(
        summary_df.iloc[
            0
        ][
            "transition_weight"
        ]
    )

    best_boost = float(
        summary_df.iloc[
            0
        ][
            "n1_emission_boost"
        ]
    )

    return (
        best_weight,
        best_boost,
    )


# ============================================================
# 12. Evaluate one feature configuration
# ============================================================

def run_nested_loso_configuration(
    dataset,
    feature_columns,
    configuration_name,
):

    fold_rows = []
    epoch_frames = []

    print("\n" + "#" * 100)
    print(
        "CONFIGURATION:",
        configuration_name
    )
    print(
        "Feature count:",
        len(
            feature_columns
        )
    )
    print("#" * 100)

    for outer_idx, test_subject in enumerate(
        subjects,
        start=1,
    ):

        print(
            f"\nOuter fold "
            f"{outer_idx}/{len(subjects)} "
            f"- {test_subject}"
        )

        train_raw = dataset[
            dataset[
                "subject"
            ] != test_subject
        ].copy()

        test_raw = dataset[
            dataset[
                "subject"
            ] == test_subject
        ].copy()

        outer_train_subjects = sorted(
            train_raw[
                "subject"
            ].unique()
        )

        inner_start = time.perf_counter()

        (
            selected_weight,
            selected_boost,
        ) = select_parameters_inner_loso(
            train_raw,
            outer_train_subjects,
            feature_columns,
        )

        inner_time = (
            time.perf_counter()
            - inner_start
        )

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

        model = make_model()

        train_start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - train_start
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        aligned_probabilities = (
            align_model_probabilities(
                model,
                probabilities,
                stage_order,
            )
        )

        (
            start_probabilities,
            transition_probabilities,
        ) = estimate_temporal_priors(
            train_raw,
            stage_order,
            laplace=1.0,
        )

        metadata_df = test_raw[
            [
                "subject",
                "recording",
                "epoch",
                "start_sec",
                "label",
            ]
        ].copy()

        predictions = (
            viterbi_decode_by_recording(
                metadata_df,
                aligned_probabilities,
                start_probabilities,
                transition_probabilities,
                selected_weight,
                selected_boost,
                stage_order,
            )
        )

        overall = overall_metrics(
            y_test,
            predictions,
        )

        per_class = per_class_metrics(
            y_test,
            predictions,
            stage_order,
        )

        row = {
            "configuration":
                configuration_name,

            "test_subject":
                test_subject,

            "feature_count":
                len(
                    feature_columns
                ),

            "selected_transition_weight":
                selected_weight,

            "selected_n1_emission_boost":
                selected_boost,

            "training_time_sec":
                training_time,

            "inner_selection_time_sec":
                inner_time,

            "accuracy":
                overall[
                    "accuracy"
                ],

            "balanced_accuracy":
                overall[
                    "balanced_accuracy"
                ],

            "macro_f1":
                overall[
                    "macro_f1"
                ],
        }

        for stage in stage_order:

            row[
                f"{stage}_f1"
            ] = per_class[
                stage
            ][
                "f1"
            ]

        fold_rows.append(
            row
        )

        epoch_df = metadata_df.copy()

        epoch_df[
            "configuration"
        ] = configuration_name

        epoch_df[
            "prediction"
        ] = predictions

        epoch_frames.append(
            epoch_df
        )

        print(
            "  Macro F1:",
            round(
                overall[
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

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.concat(
            epoch_frames,
            ignore_index=True,
        ),
    )


# ============================================================
# 13. Run base and augmented configurations
# ============================================================

(
    base_fold_df,
    base_epoch_df,
) = run_nested_loso_configuration(
    df_original,
    base_feature_columns,
    "stage40_base_features",
)

(
    augmented_fold_df,
    augmented_epoch_df,
) = run_nested_loso_configuration(
    df_augmented,
    augmented_feature_columns,
    "stage41_augmented_features",
)


fold_results_df = pd.concat(
    [
        base_fold_df,
        augmented_fold_df,
    ],
    ignore_index=True,
)

epoch_predictions_df = pd.concat(
    [
        base_epoch_df,
        augmented_epoch_df,
    ],
    ignore_index=True,
)


# ============================================================
# 14. Aggregate comparison
# ============================================================

aggregate_rows = []


for configuration_name in [
    "stage40_base_features",
    "stage41_augmented_features",
]:

    fold_subset = fold_results_df[
        fold_results_df[
            "configuration"
        ] == configuration_name
    ]

    epoch_subset = epoch_predictions_df[
        epoch_predictions_df[
            "configuration"
        ] == configuration_name
    ]

    pooled_overall = overall_metrics(
        epoch_subset[
            "label"
        ],
        epoch_subset[
            "prediction"
        ],
    )

    pooled_class = per_class_metrics(
        epoch_subset[
            "label"
        ],
        epoch_subset[
            "prediction"
        ],
        stage_order,
    )

    row = {
        "configuration":
            configuration_name,

        "feature_count":
            int(
                fold_subset[
                    "feature_count"
                ].iloc[
                    0
                ]
            ),

        "mean_subject_macro_f1":
            float(
                fold_subset[
                    "macro_f1"
                ].mean()
            ),

        "std_subject_macro_f1":
            float(
                fold_subset[
                    "macro_f1"
                ].std(
                    ddof=0
                )
            ),

        "min_subject_macro_f1":
            float(
                fold_subset[
                    "macro_f1"
                ].min()
            ),

        "max_subject_macro_f1":
            float(
                fold_subset[
                    "macro_f1"
                ].max()
            ),

        "pooled_macro_f1":
            pooled_overall[
                "macro_f1"
            ],

        "pooled_balanced_accuracy":
            pooled_overall[
                "balanced_accuracy"
            ],
    }

    for stage in stage_order:

        row[
            f"{stage}_f1"
        ] = pooled_class[
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


base_row = aggregate_df[
    aggregate_df[
        "configuration"
    ] == "stage40_base_features"
].iloc[
    0
]


aggregate_df[
    "delta_mean_subject_macro_f1"
] = (
    aggregate_df[
        "mean_subject_macro_f1"
    ]
    - float(
        base_row[
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
        base_row[
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
        base_row[
            "N1_f1"
        ]
    )
)


# ============================================================
# 15. Subject-level comparison
# ============================================================

base_subject = (
    base_fold_df[
        [
            "test_subject",
            "macro_f1",
            "N1_f1",
        ]
    ]
    .rename(
        columns={
            "macro_f1":
                "base_macro_f1",

            "N1_f1":
                "base_N1_f1",
        }
    )
)

aug_subject = (
    augmented_fold_df[
        [
            "test_subject",
            "macro_f1",
            "N1_f1",
        ]
    ]
    .rename(
        columns={
            "macro_f1":
                "augmented_macro_f1",

            "N1_f1":
                "augmented_N1_f1",
        }
    )
)

subject_comparison_df = (
    base_subject
    .merge(
        aug_subject,
        on="test_subject",
        how="inner",
    )
)

subject_comparison_df[
    "macro_f1_delta"
] = (
    subject_comparison_df[
        "augmented_macro_f1"
    ]
    - subject_comparison_df[
        "base_macro_f1"
    ]
)

subject_comparison_df[
    "N1_f1_delta"
] = (
    subject_comparison_df[
        "augmented_N1_f1"
    ]
    - subject_comparison_df[
        "base_N1_f1"
    ]
)


# ============================================================
# 16. Print comparison
# ============================================================

print("\n" + "=" * 100)
print("STAGE 41 FEATURE AUGMENTATION COMPARISON")
print("=" * 100)

display_columns = [
    "configuration",
    "feature_count",
    "mean_subject_macro_f1",
    "pooled_macro_f1",
    "min_subject_macro_f1",
    "Wake_f1",
    "N1_f1",
    "N2_f1",
    "N3_f1",
    "REM_f1",
    "delta_mean_subject_macro_f1",
    "delta_N1_f1",
]

print(
    aggregate_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print("\n" + "=" * 100)
print("SUBJECT-LEVEL COMPARISON")
print("=" * 100)

print(
    subject_comparison_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 17. Final summary
# ============================================================

augmented_row = aggregate_df[
    aggregate_df[
        "configuration"
    ] == "stage41_augmented_features"
].iloc[
    0
]

mean_macro_delta = float(
    augmented_row[
        "delta_mean_subject_macro_f1"
    ]
)

pooled_macro_delta = float(
    augmented_row[
        "delta_pooled_macro_f1"
    ]
)

n1_delta = float(
    augmented_row[
        "delta_N1_f1"
    ]
)

augmented_mean_macro_f1 = float(
    augmented_row[
        "mean_subject_macro_f1"
    ]
)

augmented_pooled_macro_f1 = float(
    augmented_row[
        "pooled_macro_f1"
    ]
)

augmented_n1_f1 = float(
    augmented_row[
        "N1_f1"
    ]
)


print("\n" + "=" * 100)
print("STAGE 41 SUMMARY")
print("=" * 100)

print(
    "\nBase feature count:",
    len(
        base_feature_columns
    )
)

print(
    "Derived feature count:",
    len(
        derived_feature_columns
    )
)

print(
    "Augmented feature count:",
    len(
        augmented_feature_columns
    )
)

print(
    "\nStage 40 base mean LOSO Macro F1:",
    round(
        float(
            base_row[
                "mean_subject_macro_f1"
            ]
        ),
        4,
    )
)

print(
    "Stage 41 augmented mean LOSO Macro F1:",
    round(
        augmented_mean_macro_f1,
        4,
    )
)

print(
    "Mean Macro F1 delta:",
    round(
        mean_macro_delta,
        4,
    )
)

print(
    "\nStage 40 pooled Macro F1:",
    round(
        float(
            base_row[
                "pooled_macro_f1"
            ]
        ),
        4,
    )
)

print(
    "Stage 41 pooled Macro F1:",
    round(
        augmented_pooled_macro_f1,
        4,
    )
)

print(
    "Pooled Macro F1 delta:",
    round(
        pooled_macro_delta,
        4,
    )
)

print(
    "\nStage 40 pooled N1 F1:",
    round(
        float(
            base_row[
                "N1_f1"
            ]
        ),
        4,
    )
)

print(
    "Stage 41 pooled N1 F1:",
    round(
        augmented_n1_f1,
        4,
    )
)

print(
    "N1 F1 delta:",
    round(
        n1_delta,
        4,
    )
)


print("\nInterpretation:")

if (
    mean_macro_delta > 0.01
    and pooled_macro_delta > 0
):

    print(
        "STRONG IMPROVEMENT:"
    )

    print(
        "Causal temporal dynamic features provide "
        "meaningful cross-subject improvement."
    )

    print(
        "Stage 41 augmented features should advance "
        "to feature ablation and compactness testing."
    )

elif mean_macro_delta > 0:

    print(
        "MARGINAL IMPROVEMENT:"
    )

    print(
        "Temporal dynamic features provide a small benefit."
    )

    print(
        "Stage 42 should determine which derived features "
        "are actually responsible for the gain."
    )

else:

    print(
        "NO OVERALL IMPROVEMENT:"
    )

    print(
        "The Stage 40 feature set remains stronger."
    )

    print(
        "Do not replace the Stage 40 canonical candidate."
    )


# ============================================================
# 18. Save outputs
# ============================================================

fold_results_path = (
    results_dir
    / "stage41_fold_results.csv"
)

aggregate_path = (
    results_dir
    / "stage41_feature_augmentation_benchmark.csv"
)

subject_comparison_path = (
    results_dir
    / "stage41_subject_comparison.csv"
)

derived_features_path = (
    results_dir
    / "stage41_derived_feature_list.csv"
)

epoch_predictions_path = (
    results_dir
    / "stage41_epoch_predictions.csv"
)

summary_path = (
    results_dir
    / "stage41_summary.csv"
)


fold_results_df.to_csv(
    fold_results_path,
    index=False,
)

aggregate_df.to_csv(
    aggregate_path,
    index=False,
)

subject_comparison_df.to_csv(
    subject_comparison_path,
    index=False,
)

pd.DataFrame(
    {
        "derived_feature":
            derived_feature_columns
    }
).to_csv(
    derived_features_path,
    index=False,
)

epoch_predictions_df.to_csv(
    epoch_predictions_path,
    index=False,
)

pd.DataFrame([
    {
        "base_feature_count":
            len(
                base_feature_columns
            ),

        "derived_feature_count":
            len(
                derived_feature_columns
            ),

        "augmented_feature_count":
            len(
                augmented_feature_columns
            ),

        "base_mean_loso_macro_f1":
            float(
                base_row[
                    "mean_subject_macro_f1"
                ]
            ),

        "augmented_mean_loso_macro_f1":
            augmented_mean_macro_f1,

        "mean_macro_f1_delta":
            mean_macro_delta,

        "base_pooled_macro_f1":
            float(
                base_row[
                    "pooled_macro_f1"
                ]
            ),

        "augmented_pooled_macro_f1":
            augmented_pooled_macro_f1,

        "pooled_macro_f1_delta":
            pooled_macro_delta,

        "base_pooled_n1_f1":
            float(
                base_row[
                    "N1_f1"
                ]
            ),

        "augmented_pooled_n1_f1":
            augmented_n1_f1,

        "n1_f1_delta":
            n1_delta,
    }
]).to_csv(
    summary_path,
    index=False,
)


print("\nSaved outputs:")

for path in [
    fold_results_path,
    aggregate_path,
    subject_comparison_path,
    derived_features_path,
    epoch_predictions_path,
    summary_path,
]:

    print(
        path
    )


print("\n" + "=" * 100)
print("STAGE 41 COMPLETE")
print("=" * 100)
