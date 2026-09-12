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
# Stage 42
# Nested Derived-Feature Selection / Ablation
#
# Motivation:
# Stage 41:
#   Base feature count       = 45
#   Derived feature count    = 60
#   Augmented feature count  = 105
#
#   Full augmentation hurt performance.
#
# Hypothesis:
#   Only a small subset of derived temporal features is useful.
#   The rest may add redundancy/noise.
#
# Rigorous nested design:
#
# OUTER LOSO:
#   one completely unseen subject = final test
#
# INNER LOSO:
#   using only outer-training subjects:
#
#   1. Rank DERIVED features using N1-vs-nonN1
#      Random Forest feature importance on inner-training data.
#
#   2. Test top-k derived feature counts:
#         0, 5, 10, 15, 20
#
#   3. For each top-k, jointly search:
#         transition_weight
#         N1 emission boost
#
#   4. Select the best combination using
#         mean Macro F1
#         then mean N1 F1
#         then mean minimum-class F1
#
# OUTER TRAIN:
#   re-rank derived features using all outer-training subjects,
#   keep selected top-k,
#   train final model,
#   test on unseen outer subject.
#
# This avoids selecting derived features from the outer test subject.
# ============================================================


print("=" * 104)
print("STAGE 42 - NESTED DERIVED-FEATURE SELECTION / ABLATION")
print("=" * 104)


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
# 2. Recreate Stage 41 causal derived features
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


print("\nDerived feature count:")
print(len(derived_feature_columns))


# ============================================================
# 3. Candidate top-k and temporal parameters
# ============================================================

top_k_candidates = [
    0,
    5,
    10,
    15,
    20,
]

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


print("\nTop-k candidates:")
print(top_k_candidates)


# ============================================================
# 4. Normalization
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
# 5. RF factories
# ============================================================

def make_multiclass_model():

    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


def make_n1_ranking_model():

    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


# ============================================================
# 6. Metrics
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
# 7. Rank derived features using TRAINING DATA ONLY
#
# N1 vs non-N1 feature importance.
# ============================================================

def rank_derived_features_for_n1(
    train_df,
    derived_features,
):

    if len(
        derived_features
    ) == 0:

        return []

    means, stds = (
        calculate_training_statistics(
            train_df,
            derived_features,
        )
    )

    normalized = (
        normalize_with_fixed_statistics(
            train_df,
            derived_features,
            means,
            stds,
        )
    )

    X = normalized[
        derived_features
    ]

    y = (
        normalized[
            "label"
        ] == "N1"
    ).astype(int)

    model = make_n1_ranking_model()

    model.fit(
        X,
        y,
    )

    ranking_df = pd.DataFrame(
        {
            "feature":
                derived_features,

            "importance":
                model.feature_importances_,
        }
    )

    ranking_df = (
        ranking_df
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return ranking_df[
        "feature"
    ].tolist()


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
# 9. Probability alignment
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
# 11. Inner LOSO:
# select top-k + transition weight + N1 boost
# ============================================================

def select_configuration_inner_loso(
    outer_train_df,
    outer_train_subjects,
):

    configuration_records = {}

    for top_k in top_k_candidates:

        for weight in transition_weight_candidates:

            for boost in n1_emission_boost_candidates:

                configuration_records[
                    (
                        top_k,
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

        # ----------------------------------------------------
        # Derived-feature ranking from INNER TRAINING ONLY
        # ----------------------------------------------------

        ranked_derived = (
            rank_derived_features_for_n1(
                inner_train_raw,
                derived_feature_columns,
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

        y_test = inner_test_raw[
            "label"
        ]

        # ----------------------------------------------------
        # One RF per top-k.
        # Temporal grid reuses same probabilities.
        # ----------------------------------------------------

        for top_k in top_k_candidates:

            selected_derived = (
                ranked_derived[
                    :top_k
                ]
                if top_k > 0
                else []
            )

            feature_columns = (
                base_feature_columns
                + selected_derived
            )

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

            model = make_multiclass_model()

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

                    configuration_records[
                        (
                            top_k,
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

    summary_rows = []

    for (
        top_k,
        weight,
        boost,
    ), records in configuration_records.items():

        temp_df = pd.DataFrame(
            records
        )

        summary_rows.append(
            {
                "top_k":
                    top_k,

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

    best_row = summary_df.iloc[
        0
    ]

    return (
        int(
            best_row[
                "top_k"
            ]
        ),
        float(
            best_row[
                "transition_weight"
            ]
        ),
        float(
            best_row[
                "n1_emission_boost"
            ]
        ),
        summary_df,
    )


# ============================================================
# 12. Outer LOSO
# ============================================================

outer_fold_rows = []
inner_search_rows = []
selected_feature_rows = []
epoch_frames = []


print("\n" + "=" * 104)
print("STARTING OUTER LOSO")
print("=" * 104)


for outer_idx, test_subject in enumerate(
    subjects,
    start=1,
):

    print("\n" + "#" * 104)

    print(
        f"OUTER FOLD "
        f"{outer_idx}/{len(subjects)} "
        f"- TEST: "
        f"{test_subject}"
    )

    print("#" * 104)

    train_raw = df_augmented[
        df_augmented[
            "subject"
        ] != test_subject
    ].copy()

    test_raw = df_augmented[
        df_augmented[
            "subject"
        ] == test_subject
    ].copy()

    outer_train_subjects = sorted(
        train_raw[
            "subject"
        ].unique()
    )

    # --------------------------------------------------------
    # Nested selection
    # --------------------------------------------------------

    inner_start = time.perf_counter()

    (
        selected_top_k,
        selected_weight,
        selected_boost,
        inner_summary_df,
    ) = select_configuration_inner_loso(
        train_raw,
        outer_train_subjects,
    )

    inner_time_sec = (
        time.perf_counter()
        - inner_start
    )

    print(
        "\nSelected top-k:",
        selected_top_k
    )

    print(
        "Selected transition weight:",
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

                "top_k":
                    int(
                        row[
                            "top_k"
                        ]
                    ),

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

                "mean_balanced_accuracy":
                    float(
                        row[
                            "mean_balanced_accuracy"
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
                        int(
                            row[
                                "top_k"
                            ]
                        )
                        == selected_top_k
                        and float(
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
    # Re-rank derived features using ALL OUTER TRAINING ONLY
    # --------------------------------------------------------

    ranked_derived = (
        rank_derived_features_for_n1(
            train_raw,
            derived_feature_columns,
        )
    )

    selected_derived = (
        ranked_derived[
            :selected_top_k
        ]
        if selected_top_k > 0
        else []
    )

    final_feature_columns = (
        base_feature_columns
        + selected_derived
    )

    for rank_position, feature in enumerate(
        selected_derived,
        start=1,
    ):

        selected_feature_rows.append(
            {
                "outer_test_subject":
                    test_subject,

                "rank":
                    rank_position,

                "feature":
                    feature,
            }
        )

    # --------------------------------------------------------
    # Final outer model
    # --------------------------------------------------------

    means, stds = (
        calculate_training_statistics(
            train_raw,
            final_feature_columns,
        )
    )

    train_normalized = (
        normalize_with_fixed_statistics(
            train_raw,
            final_feature_columns,
            means,
            stds,
        )
    )

    test_normalized = (
        normalize_with_fixed_statistics(
            test_raw,
            final_feature_columns,
            means,
            stds,
        )
    )

    X_train = train_normalized[
        final_feature_columns
    ]

    y_train = train_normalized[
        "label"
    ]

    X_test = test_normalized[
        final_feature_columns
    ]

    y_test = test_normalized[
        "label"
    ]

    model = make_multiclass_model()

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
        "test_subject":
            test_subject,

        "selected_top_k":
            selected_top_k,

        "selected_transition_weight":
            selected_weight,

        "selected_n1_emission_boost":
            selected_boost,

        "feature_count":
            len(
                final_feature_columns
            ),

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

    epoch_df = metadata_df.copy()

    epoch_df[
        "prediction"
    ] = predictions

    epoch_df[
        "selected_top_k"
    ] = selected_top_k

    epoch_df[
        "selected_transition_weight"
    ] = selected_weight

    epoch_df[
        "selected_n1_emission_boost"
    ] = selected_boost

    epoch_frames.append(
        epoch_df
    )

    print(
        "\nOuter Macro F1:",
        round(
            overall[
                "macro_f1"
            ],
            4,
        )
    )

    print(
        "Outer N1 F1:",
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
# 13. Build result tables
# ============================================================

fold_results_df = pd.DataFrame(
    outer_fold_rows
)

inner_search_df = pd.DataFrame(
    inner_search_rows
)

selected_features_df = pd.DataFrame(
    selected_feature_rows
)

epoch_predictions_df = pd.concat(
    epoch_frames,
    ignore_index=True,
)


# ============================================================
# 14. Aggregate Stage 42 results
# ============================================================

mean_macro_f1 = float(
    fold_results_df[
        "macro_f1"
    ].mean()
)

std_macro_f1 = float(
    fold_results_df[
        "macro_f1"
    ].std(
        ddof=0
    )
)

min_macro_f1 = float(
    fold_results_df[
        "macro_f1"
    ].min()
)

max_macro_f1 = float(
    fold_results_df[
        "macro_f1"
    ].max()
)

mean_balanced_accuracy = float(
    fold_results_df[
        "balanced_accuracy"
    ].mean()
)

pooled_overall = overall_metrics(
    epoch_predictions_df[
        "label"
    ],
    epoch_predictions_df[
        "prediction"
    ],
)

pooled_class = per_class_metrics(
    epoch_predictions_df[
        "label"
    ],
    epoch_predictions_df[
        "prediction"
    ],
    stage_order,
)


# ============================================================
# 15. Stage 40 reference
# ============================================================

STAGE40_MEAN_MACRO_F1 = 0.6902
STAGE40_POOLED_MACRO_F1 = 0.6878
STAGE40_POOLED_N1_F1 = 0.3855


delta_vs_stage40_mean = (
    mean_macro_f1
    - STAGE40_MEAN_MACRO_F1
)

delta_vs_stage40_pooled = (
    pooled_overall[
        "macro_f1"
    ]
    - STAGE40_POOLED_MACRO_F1
)

delta_vs_stage40_n1 = (
    pooled_class[
        "N1"
    ][
        "f1"
    ]
    - STAGE40_POOLED_N1_F1
)


# ============================================================
# 16. Top-k selection distribution
# ============================================================

top_k_distribution_df = (
    fold_results_df[
        "selected_top_k"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "top_k"
    )
    .reset_index(
        name="outer_fold_count"
    )
)


# ============================================================
# 17. Most frequently selected derived features
# ============================================================

if len(
    selected_features_df
) > 0:

    feature_frequency_df = (
        selected_features_df[
            "feature"
        ]
        .value_counts()
        .rename_axis(
            "feature"
        )
        .reset_index(
            name="selected_fold_count"
        )
    )

else:

    feature_frequency_df = pd.DataFrame(
        columns=[
            "feature",
            "selected_fold_count",
        ]
    )


# ============================================================
# 18. Print results
# ============================================================

print("\n" + "=" * 104)
print("STAGE 42 OUTER-FOLD RESULTS")
print("=" * 104)

print(
    fold_results_df[
        [
            "test_subject",
            "selected_top_k",
            "selected_transition_weight",
            "selected_n1_emission_boost",
            "feature_count",
            "macro_f1",
            "N1_f1",
            "Wake_f1",
            "N2_f1",
            "N3_f1",
            "REM_f1",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print("\n" + "=" * 104)
print("TOP-K SELECTION DISTRIBUTION")
print("=" * 104)

print(
    top_k_distribution_df.to_string(
        index=False
    )
)


print("\n" + "=" * 104)
print("MOST FREQUENTLY SELECTED DERIVED FEATURES")
print("=" * 104)

if len(
    feature_frequency_df
) > 0:

    print(
        feature_frequency_df
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "No derived features were selected."
    )


# ============================================================
# 19. Final summary
# ============================================================

print("\n" + "=" * 104)
print("STAGE 42 SUMMARY")
print("=" * 104)

print(
    "\nMean LOSO Macro F1:",
    round(
        mean_macro_f1,
        4,
    )
)

print(
    "Std LOSO Macro F1:",
    round(
        std_macro_f1,
        4,
    )
)

print(
    "Min subject Macro F1:",
    round(
        min_macro_f1,
        4,
    )
)

print(
    "Max subject Macro F1:",
    round(
        max_macro_f1,
        4,
    )
)

print(
    "\nPooled Macro F1:",
    round(
        pooled_overall[
            "macro_f1"
        ],
        4,
    )
)

print(
    "Pooled Balanced Accuracy:",
    round(
        pooled_overall[
            "balanced_accuracy"
        ],
        4,
    )
)

print(
    "\nPooled per-class F1:"
)

for stage in stage_order:

    print(
        f"  {stage}:",
        round(
            pooled_class[
                stage
            ][
                "f1"
            ],
            4,
        )
    )


print("\nStage 40 reference:")
print(
    "  Mean LOSO Macro F1:",
    STAGE40_MEAN_MACRO_F1
)

print(
    "  Pooled Macro F1:",
    STAGE40_POOLED_MACRO_F1
)

print(
    "  Pooled N1 F1:",
    STAGE40_POOLED_N1_F1
)


print("\nDelta vs Stage 40:")

print(
    "  Mean Macro F1 delta:",
    round(
        delta_vs_stage40_mean,
        4,
    )
)

print(
    "  Pooled Macro F1 delta:",
    round(
        delta_vs_stage40_pooled,
        4,
    )
)

print(
    "  N1 F1 delta:",
    round(
        delta_vs_stage40_n1,
        4,
    )
)


print("\nInterpretation:")

if (
    delta_vs_stage40_mean > 0.01
    and delta_vs_stage40_pooled > 0
):

    print(
        "STRONG IMPROVEMENT:"
    )

    print(
        "Nested derived-feature selection beats Stage 40."
    )

    print(
        "Advance Stage 42 to final canonical model selection."
    )

elif (
    delta_vs_stage40_mean > 0
    and delta_vs_stage40_pooled >= 0
):

    print(
        "MARGINAL IMPROVEMENT:"
    )

    print(
        "Selected derived features provide a small gain."
    )

    print(
        "Keep Stage 42 for final comparison, but verify "
        "latency and compactness before replacing Stage 40."
    )

else:

    print(
        "NO OVERALL IMPROVEMENT:"
    )

    print(
        "Stage 40 remains the stronger canonical candidate."
    )

    print(
        "Do not keep the derived-feature branch unless "
        "it offers another practical advantage."
    )


# ============================================================
# 20. Save outputs
# ============================================================

fold_results_path = (
    results_dir
    / "stage42_fold_results.csv"
)

inner_search_path = (
    results_dir
    / "stage42_inner_configuration_search.csv"
)

selected_features_path = (
    results_dir
    / "stage42_selected_features_by_fold.csv"
)

feature_frequency_path = (
    results_dir
    / "stage42_feature_selection_frequency.csv"
)

top_k_distribution_path = (
    results_dir
    / "stage42_top_k_distribution.csv"
)

epoch_predictions_path = (
    results_dir
    / "stage42_epoch_predictions.csv"
)

summary_path = (
    results_dir
    / "stage42_summary.csv"
)


fold_results_df.to_csv(
    fold_results_path,
    index=False,
)

inner_search_df.to_csv(
    inner_search_path,
    index=False,
)

selected_features_df.to_csv(
    selected_features_path,
    index=False,
)

feature_frequency_df.to_csv(
    feature_frequency_path,
    index=False,
)

top_k_distribution_df.to_csv(
    top_k_distribution_path,
    index=False,
)

epoch_predictions_df.to_csv(
    epoch_predictions_path,
    index=False,
)


pd.DataFrame([
    {
        "mean_loso_macro_f1":
            mean_macro_f1,

        "std_loso_macro_f1":
            std_macro_f1,

        "min_subject_macro_f1":
            min_macro_f1,

        "max_subject_macro_f1":
            max_macro_f1,

        "pooled_macro_f1":
            pooled_overall[
                "macro_f1"
            ],

        "pooled_balanced_accuracy":
            pooled_overall[
                "balanced_accuracy"
            ],

        "pooled_wake_f1":
            pooled_class[
                "Wake"
            ][
                "f1"
            ],

        "pooled_n1_f1":
            pooled_class[
                "N1"
            ][
                "f1"
            ],

        "pooled_n2_f1":
            pooled_class[
                "N2"
            ][
                "f1"
            ],

        "pooled_n3_f1":
            pooled_class[
                "N3"
            ][
                "f1"
            ],

        "pooled_rem_f1":
            pooled_class[
                "REM"
            ][
                "f1"
            ],

        "delta_vs_stage40_mean_macro_f1":
            delta_vs_stage40_mean,

        "delta_vs_stage40_pooled_macro_f1":
            delta_vs_stage40_pooled,

        "delta_vs_stage40_n1_f1":
            delta_vs_stage40_n1,
    }
]).to_csv(
    summary_path,
    index=False,
)


print("\nSaved outputs:")

for path in [
    fold_results_path,
    inner_search_path,
    selected_features_path,
    feature_frequency_path,
    top_k_distribution_path,
    epoch_predictions_path,
    summary_path,
]:

    print(
        path
    )


print("\n" + "=" * 104)
print("STAGE 42 COMPLETE")
print("=" * 104)
