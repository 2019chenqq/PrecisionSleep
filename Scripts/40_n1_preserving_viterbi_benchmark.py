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
# Stage 40
# N1-Preserving Nested Viterbi Benchmark
#
# Motivation:
# Stage 39:
#   - Viterbi improved overall LOSO Macro F1
#   - Wake / N2 / REM improved
#   - N1 F1 decreased
#
# Hypothesis:
# Standard transition priors over-suppress short/transitional N1.
#
# Strategy:
# Add an N1 emission boost INSIDE Viterbi decoding.
#
# score_t(state) =
#     log emission(state)
#     + transition_weight * log transition(prev -> state)
#     + log(n1_emission_boost), only when state == N1
#
# Nested LOSO:
#   OUTER LOSO:
#       unseen test subject
#
#   INNER LOSO:
#       choose transition_weight
#       choose n1_emission_boost
#       using ONLY outer-training subjects
#
# Compare:
#   1. baseline_argmax
#   2. standard_viterbi
#   3. n1_preserving_viterbi
# ============================================================


print("=" * 98)
print("STAGE 40 - N1-PRESERVING NESTED VITERBI BENCHMARK")
print("=" * 98)


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
        df_raw[
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

print("\nFeature count:")
print(len(feature_columns))

print("\nTotal epochs:")
print(len(df_raw))

print("\nStage order:")
print(stage_order)


# ============================================================
# 2. Search grid
#
# Kept deliberately small so Stage 40 remains practical.
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

print("\nTransition weights:")
print(transition_weight_candidates)

print("\nN1 emission boosts:")
print(n1_emission_boost_candidates)


# ============================================================
# 3. Training-global normalization
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
# 4. RF
# ============================================================

def make_model():

    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


# ============================================================
# 5. Metrics
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
# 6. Temporal priors
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
# 7. Probability alignment
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
# 8. Viterbi with optional N1 emission boost
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

    if (
        n1_emission_boost
        > 0
    ):

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

    best_path = np.zeros(
        n_epochs,
        dtype=int,
    )

    best_path[
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

        best_path[
            t
        ] = backpointer[
            t + 1,
            best_path[
                t + 1
            ],
        ]

    return best_path


# ============================================================
# 9. Recording-wise decoding
# ============================================================

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
# 10. Inner nested parameter search
# ============================================================

def select_parameters_inner_loso(
    outer_train_raw,
    outer_train_subjects,
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

        inner_train_raw = outer_train_raw[
            outer_train_raw[
                "subject"
            ] != inner_test_subject
        ].copy()

        inner_test_raw = outer_train_raw[
            outer_train_raw[
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
                        "test_subject":
                            inner_test_subject,

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

    summary_rows = []

    for (
        weight,
        boost,
    ), records in parameter_records.items():

        temp_df = pd.DataFrame(
            records
        )

        summary_rows.append(
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

                "mean_balanced_accuracy":
                    float(
                        temp_df[
                            "balanced_accuracy"
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

                "std_macro_f1":
                    float(
                        temp_df[
                            "macro_f1"
                        ].std(
                            ddof=0
                        )
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # Primary objective:
    # overall cross-subject Macro F1
    #
    # Secondary:
    # N1 F1, then minimum class F1
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
        summary_df,
    )


# ============================================================
# 11. Outer LOSO
# ============================================================

outer_fold_rows = []
inner_search_rows = []
epoch_frames = []


print("\n" + "=" * 98)
print("STARTING OUTER LOSO")
print("=" * 98)


for outer_idx, test_subject in enumerate(
    subjects,
    start=1,
):

    print("\n" + "#" * 98)

    print(
        f"OUTER FOLD "
        f"{outer_idx}/{len(subjects)} "
        f"- TEST: "
        f"{test_subject}"
    )

    print("#" * 98)

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

    outer_train_subjects = sorted(
        train_raw[
            "subject"
        ].unique()
    )

    # --------------------------------------------------------
    # Inner nested parameter search
    # --------------------------------------------------------

    inner_start = time.perf_counter()

    (
        selected_weight,
        selected_boost,
        inner_summary_df,
    ) = select_parameters_inner_loso(
        train_raw,
        outer_train_subjects,
    )

    inner_time_sec = (
        time.perf_counter()
        - inner_start
    )

    print(
        "\nSelected transition weight:",
        selected_weight
    )

    print(
        "Selected N1 emission boost:",
        selected_boost
    )

    for _, row in inner_summary_df.iterrows():

        inner_search_rows.append(
            {
                "outer_test_subject":
                    test_subject,

                "transition_weight":
                    float(
                        row[
                            "transition_weight"
                        ]
                    ),

                "n1_emission_boost":
                    float(
                        row[
                            "n1_emission_boost"
                        ]
                    ),

                "mean_macro_f1":
                    float(
                        row[
                            "mean_macro_f1"
                        ]
                    ),

                "mean_balanced_accuracy":
                    float(
                        row[
                            "mean_balanced_accuracy"
                        ]
                    ),

                "mean_n1_f1":
                    float(
                        row[
                            "mean_n1_f1"
                        ]
                    ),

                "mean_min_class_f1":
                    float(
                        row[
                            "mean_min_class_f1"
                        ]
                    ),

                "std_macro_f1":
                    float(
                        row[
                            "std_macro_f1"
                        ]
                    ),

                "selected":
                    bool(
                        float(
                            row[
                                "transition_weight"
                            ]
                        )
                        == selected_weight
                        and float(
                            row[
                                "n1_emission_boost"
                            ]
                        )
                        == selected_boost
                    ),
            }
        )

    # --------------------------------------------------------
    # Train final outer RF
    # --------------------------------------------------------

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

    training_time_sec = (
        time.perf_counter()
        - train_start
    )

    probabilities = model.predict_proba(
        X_test
    )

    aligned_probabilities = (
        align_model_probabilities(
            model,
            probabilities,
            stage_order,
        )
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

    # --------------------------------------------------------
    # Baseline argmax
    # --------------------------------------------------------

    class_array = np.array(
        model.classes_,
        dtype=object,
    )

    baseline_predictions = class_array[
        np.argmax(
            probabilities,
            axis=1,
        )
    ]

    # --------------------------------------------------------
    # Temporal priors from outer training data
    # --------------------------------------------------------

    (
        start_probabilities,
        transition_probabilities,
    ) = estimate_temporal_priors(
        train_raw,
        stage_order,
        laplace=1.0,
    )

    # --------------------------------------------------------
    # Standard Viterbi
    #
    # same selected temporal weight,
    # boost fixed at 1.0
    # --------------------------------------------------------

    standard_viterbi_predictions = (
        viterbi_decode_by_recording(
            metadata_df,
            aligned_probabilities,
            start_probabilities,
            transition_probabilities,
            selected_weight,
            1.0,
            stage_order,
        )
    )

    # --------------------------------------------------------
    # N1-preserving Viterbi
    # --------------------------------------------------------

    n1_preserving_predictions = (
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    strategies = {
        "baseline_argmax":
            baseline_predictions,

        "standard_viterbi":
            standard_viterbi_predictions,

        "n1_preserving_viterbi":
            n1_preserving_predictions,
    }

    for strategy_name, predictions in strategies.items():

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
            "test_subject":
                test_subject,

            "strategy":
                strategy_name,

            "selected_transition_weight":
                selected_weight,

            "selected_n1_emission_boost":
                selected_boost,

            "training_time_sec":
                training_time_sec,

            "inner_selection_time_sec":
                inner_time_sec,

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
                f"{stage}_precision"
            ] = per_class[
                stage
            ][
                "precision"
            ]

            row[
                f"{stage}_recall"
            ] = per_class[
                stage
            ][
                "recall"
            ]

            row[
                f"{stage}_f1"
            ] = per_class[
                stage
            ][
                "f1"
            ]

        outer_fold_rows.append(
            row
        )

    # --------------------------------------------------------
    # Epoch-level output
    # --------------------------------------------------------

    epoch_df = metadata_df.copy()

    epoch_df[
        "selected_transition_weight"
    ] = selected_weight

    epoch_df[
        "selected_n1_emission_boost"
    ] = selected_boost

    epoch_df[
        "baseline_argmax"
    ] = baseline_predictions

    epoch_df[
        "standard_viterbi"
    ] = standard_viterbi_predictions

    epoch_df[
        "n1_preserving_viterbi"
    ] = n1_preserving_predictions

    epoch_df[
        "standard_viterbi_changed"
    ] = (
        standard_viterbi_predictions
        != baseline_predictions
    )

    epoch_df[
        "n1_preserving_changed"
    ] = (
        n1_preserving_predictions
        != baseline_predictions
    )

    epoch_frames.append(
        epoch_df
    )

    # --------------------------------------------------------
    # Fold print
    # --------------------------------------------------------

    baseline_metrics = overall_metrics(
        y_test,
        baseline_predictions,
    )

    standard_metrics = overall_metrics(
        y_test,
        standard_viterbi_predictions,
    )

    preserving_metrics = overall_metrics(
        y_test,
        n1_preserving_predictions,
    )

    preserving_class = per_class_metrics(
        y_test,
        n1_preserving_predictions,
        stage_order,
    )

    print(
        "\nBaseline Macro F1:",
        round(
            baseline_metrics[
                "macro_f1"
            ],
            4,
        )
    )

    print(
        "Standard Viterbi Macro F1:",
        round(
            standard_metrics[
                "macro_f1"
            ],
            4,
        )
    )

    print(
        "N1-preserving Viterbi Macro F1:",
        round(
            preserving_metrics[
                "macro_f1"
            ],
            4,
        )
    )

    print(
        "N1-preserving N1 F1:",
        round(
            preserving_class[
                "N1"
            ][
                "f1"
            ],
            4,
        )
    )


# ============================================================
# 12. Build result tables
# ============================================================

fold_results_df = pd.DataFrame(
    outer_fold_rows
)

inner_search_df = pd.DataFrame(
    inner_search_rows
)

epoch_predictions_df = pd.concat(
    epoch_frames,
    ignore_index=True,
)


# ============================================================
# 13. Aggregate strategies
# ============================================================

aggregate_rows = []


for strategy_name in [
    "baseline_argmax",
    "standard_viterbi",
    "n1_preserving_viterbi",
]:

    subset = fold_results_df[
        fold_results_df[
            "strategy"
        ] == strategy_name
    ]

    pooled_predictions = (
        epoch_predictions_df[
            strategy_name
        ]
    )

    pooled_overall = overall_metrics(
        epoch_predictions_df[
            "label"
        ],
        pooled_predictions,
    )

    pooled_class = per_class_metrics(
        epoch_predictions_df[
            "label"
        ],
        pooled_predictions,
        stage_order,
    )

    row = {
        "strategy":
            strategy_name,

        "mean_subject_macro_f1":
            float(
                subset[
                    "macro_f1"
                ].mean()
            ),

        "std_subject_macro_f1":
            float(
                subset[
                    "macro_f1"
                ].std(
                    ddof=0
                )
            ),

        "min_subject_macro_f1":
            float(
                subset[
                    "macro_f1"
                ].min()
            ),

        "max_subject_macro_f1":
            float(
                subset[
                    "macro_f1"
                ].max()
            ),

        "mean_balanced_accuracy":
            float(
                subset[
                    "balanced_accuracy"
                ].mean()
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


# ============================================================
# 14. Deltas vs baseline
# ============================================================

baseline_row = aggregate_df[
    aggregate_df[
        "strategy"
    ] == "baseline_argmax"
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

print("\n" + "=" * 98)
print("STAGE 40 DECODING RANKING")
print("=" * 98)

display_columns = [
    "strategy",
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
# 17. Selected parameters
# ============================================================

selected_params_df = (
    fold_results_df[
        fold_results_df[
            "strategy"
        ] == "n1_preserving_viterbi"
    ][
        [
            "test_subject",
            "selected_transition_weight",
            "selected_n1_emission_boost",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        by="test_subject"
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 98)
print("SELECTED PARAMETERS")
print("=" * 98)

print(
    selected_params_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.3f}"
        ),
    )
)


# ============================================================
# 18. Best strategy summary
# ============================================================

best_row = ranking_df.iloc[
    0
]

best_strategy = str(
    best_row[
        "strategy"
    ]
)

best_mean_macro_f1 = float(
    best_row[
        "mean_subject_macro_f1"
    ]
)

best_pooled_macro_f1 = float(
    best_row[
        "pooled_macro_f1"
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


print("\n" + "=" * 98)
print("STAGE 40 SUMMARY")
print("=" * 98)

print(
    "\nBest strategy:",
    best_strategy
)

print(
    "Best mean LOSO Macro F1:",
    round(
        best_mean_macro_f1,
        4,
    )
)

print(
    "Best pooled Macro F1:",
    round(
        best_pooled_macro_f1,
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

if (
    best_strategy
    == "n1_preserving_viterbi"
    and best_macro_delta > 0.01
    and best_n1_delta >= 0
):

    print(
        "STRONG RESULT:"
    )

    print(
        "N1-preserving temporal decoding improves "
        "overall cross-subject Macro F1 without "
        "sacrificing pooled N1 F1."
    )

    print(
        "This is a strong candidate for the "
        "canonical competition pipeline."
    )

elif (
    best_strategy
    == "n1_preserving_viterbi"
    and best_macro_delta > 0
):

    print(
        "MARGINAL RESULT:"
    )

    print(
        "N1-preserving decoding provides some benefit, "
        "but the gain remains limited."
    )

elif best_strategy == "standard_viterbi":

    print(
        "STANDARD VITERBI REMAINS BEST:"
    )

    print(
        "N1 emission compensation does not provide "
        "a better overall trade-off."
    )

else:

    print(
        "BASELINE REMAINS COMPETITIVE:"
    )

    print(
        "Do not add temporal complexity unless "
        "the practical demo benefit justifies it."
    )


# ============================================================
# 19. Save outputs
# ============================================================

fold_results_path = (
    results_dir
    / "stage40_fold_results.csv"
)

inner_search_path = (
    results_dir
    / "stage40_inner_parameter_search.csv"
)

aggregate_path = (
    results_dir
    / "stage40_decoding_benchmark.csv"
)

ranking_path = (
    results_dir
    / "stage40_decoding_ranking.csv"
)

epoch_predictions_path = (
    results_dir
    / "stage40_epoch_predictions.csv"
)

selected_params_path = (
    results_dir
    / "stage40_selected_parameters.csv"
)

summary_path = (
    results_dir
    / "stage40_summary.csv"
)


fold_results_df.to_csv(
    fold_results_path,
    index=False,
)

inner_search_df.to_csv(
    inner_search_path,
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

epoch_predictions_df.to_csv(
    epoch_predictions_path,
    index=False,
)

selected_params_df.to_csv(
    selected_params_path,
    index=False,
)

pd.DataFrame([
    {
        "best_strategy":
            best_strategy,

        "best_mean_loso_macro_f1":
            best_mean_macro_f1,

        "best_pooled_macro_f1":
            best_pooled_macro_f1,

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


print("\nSaved outputs:")

for path in [
    fold_results_path,
    inner_search_path,
    aggregate_path,
    ranking_path,
    epoch_predictions_path,
    selected_params_path,
    summary_path,
]:

    print(
        path
    )


print("\n" + "=" * 98)
print("STAGE 40 COMPLETE")
print("=" * 98)
