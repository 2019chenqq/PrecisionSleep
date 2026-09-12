from pathlib import Path
import json
import time
import hashlib

import joblib
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

model_dir = (
    PROJECT_ROOT
    / "models"
    / "stage43_canonical"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Stage 43
# Final Canonical Model Selection & Lock
#
# Goal:
#   Freeze the strongest competition pipeline discovered so far.
#
# Canonical architecture:
#   - 45 original features
#   - training-global normalization
#   - RandomForestClassifier
#       n_estimators = 150
#       class_weight = balanced
#       random_state = 42
#   - N1-preserving Viterbi decoding
#
# This script:
#
#   1. Re-validates the canonical pipeline with LOSO.
#   2. Selects ONE global deployment pair:
#        transition_weight
#        n1_emission_boost
#      using subject-wise cross-validation across all subjects.
#   3. Trains the final RF using ALL available subjects.
#   4. Estimates final normalization statistics.
#   5. Estimates final start + transition probabilities.
#   6. Saves a locked deployment package.
#
# Important:
#   The final model is trained on all available subjects and is
#   therefore intended for competition/demo deployment.
#
#   Generalization claims should still use LOSO metrics,
#   NOT training-set metrics from the final fitted model.
# ============================================================


print("=" * 104)
print("STAGE 43 - FINAL CANONICAL MODEL SELECTION & LOCK")
print("=" * 104)


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
    stage
    for stage in [
        "Wake",
        "N1",
        "N2",
        "N3",
        "REM",
    ]
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

print("\nSubject count:")
print(len(subjects))

print("\nEpoch count:")
print(len(df_raw))

print("\nCanonical feature count:")
print(len(feature_columns))

print("\nStage order:")
print(stage_order)


if len(feature_columns) != 45:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 45 canonical features from Stage 40, "
        f"but found {len(feature_columns)}."
    )

    print(
        "The script will continue using the current original "
        "feature columns from multi_subject_temporal_context.csv."
    )


# ============================================================
# 2. Search grid for ONE deployment parameter pair
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
# 3. Helpers
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


def make_model():

    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


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

        if len(labels) == 0:

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
            len(labels) - 1
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


def align_model_probabilities(
    model,
    probabilities,
    stages,
):

    aligned = np.full(
        (
            len(probabilities),
            len(stages),
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
# 4. Subject-wise CV cache
#
# Train one RF per held-out subject, then reuse probabilities
# across all temporal parameter combinations.
# ============================================================

print("\n" + "=" * 104)
print("BUILDING LOSO PROBABILITY CACHE")
print("=" * 104)


fold_cache = {}


for fold_idx, test_subject in enumerate(
    subjects,
    start=1,
):

    print(
        f"Fold {fold_idx}/{len(subjects)}: "
        f"{test_subject}"
    )

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

    model = make_model()

    model.fit(
        train_normalized[
            feature_columns
        ],
        train_normalized[
            "label"
        ],
    )

    probabilities = model.predict_proba(
        test_normalized[
            feature_columns
        ]
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

    fold_cache[
        test_subject
    ] = {
        "y_true":
            test_raw[
                "label"
            ].to_numpy(),

        "metadata":
            metadata_df,

        "aligned_probabilities":
            aligned_probabilities,

        "start_probabilities":
            start_probabilities,

        "transition_probabilities":
            transition_probabilities,
    }


# ============================================================
# 5. Global deployment-parameter selection
#
# This is NOT an unbiased performance estimate.
#
# It chooses one fixed deployment configuration using all
# subject-wise CV folds.
#
# The unbiased competition validation remains Stage 40 nested LOSO.
# ============================================================

print("\n" + "=" * 104)
print("SELECTING FIXED DEPLOYMENT PARAMETERS")
print("=" * 104)


parameter_rows = []


for transition_weight in transition_weight_candidates:

    for n1_boost in n1_emission_boost_candidates:

        subject_macro_f1_values = []
        subject_n1_f1_values = []
        subject_min_class_f1_values = []
        subject_balanced_accuracy_values = []

        for test_subject in subjects:

            cached = fold_cache[
                test_subject
            ]

            predictions = (
                viterbi_decode_by_recording(
                    cached[
                        "metadata"
                    ],
                    cached[
                        "aligned_probabilities"
                    ],
                    cached[
                        "start_probabilities"
                    ],
                    cached[
                        "transition_probabilities"
                    ],
                    transition_weight,
                    n1_boost,
                    stage_order,
                )
            )

            overall = overall_metrics(
                cached[
                    "y_true"
                ],
                predictions,
            )

            per_class = per_class_metrics(
                cached[
                    "y_true"
                ],
                predictions,
                stage_order,
            )

            subject_macro_f1_values.append(
                overall[
                    "macro_f1"
                ]
            )

            subject_balanced_accuracy_values.append(
                overall[
                    "balanced_accuracy"
                ]
            )

            subject_n1_f1_values.append(
                per_class[
                    "N1"
                ][
                    "f1"
                ]
            )

            subject_min_class_f1_values.append(
                min(
                    per_class[
                        stage
                    ][
                        "f1"
                    ]
                    for stage in stage_order
                )
            )

        parameter_rows.append(
            {
                "transition_weight":
                    transition_weight,

                "n1_emission_boost":
                    n1_boost,

                "mean_subject_macro_f1":
                    float(
                        np.mean(
                            subject_macro_f1_values
                        )
                    ),

                "std_subject_macro_f1":
                    float(
                        np.std(
                            subject_macro_f1_values
                        )
                    ),

                "mean_subject_n1_f1":
                    float(
                        np.mean(
                            subject_n1_f1_values
                        )
                    ),

                "mean_min_class_f1":
                    float(
                        np.mean(
                            subject_min_class_f1_values
                        )
                    ),

                "mean_balanced_accuracy":
                    float(
                        np.mean(
                            subject_balanced_accuracy_values
                        )
                    ),
            }
        )


parameter_search_df = pd.DataFrame(
    parameter_rows
)

parameter_search_df = (
    parameter_search_df
    .sort_values(
        by=[
            "mean_subject_macro_f1",
            "mean_subject_n1_f1",
            "mean_min_class_f1",
            "mean_balanced_accuracy",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


best_parameter_row = (
    parameter_search_df.iloc[
        0
    ]
)

canonical_transition_weight = float(
    best_parameter_row[
        "transition_weight"
    ]
)

canonical_n1_emission_boost = float(
    best_parameter_row[
        "n1_emission_boost"
    ]
)


print(
    "\nSelected canonical transition weight:",
    canonical_transition_weight
)

print(
    "Selected canonical N1 emission boost:",
    canonical_n1_emission_boost
)

print(
    "CV mean Macro F1 for parameter selection:",
    round(
        float(
            best_parameter_row[
                "mean_subject_macro_f1"
            ]
        ),
        4,
    )
)


# ============================================================
# 6. Fixed-parameter LOSO report
#
# Useful for deployment consistency analysis.
# Stage 40 remains the primary nested LOSO validation.
# ============================================================

fixed_fold_rows = []
fixed_epoch_frames = []


for test_subject in subjects:

    cached = fold_cache[
        test_subject
    ]

    predictions = (
        viterbi_decode_by_recording(
            cached[
                "metadata"
            ],
            cached[
                "aligned_probabilities"
            ],
            cached[
                "start_probabilities"
            ],
            cached[
                "transition_probabilities"
            ],
            canonical_transition_weight,
            canonical_n1_emission_boost,
            stage_order,
        )
    )

    overall = overall_metrics(
        cached[
            "y_true"
        ],
        predictions,
    )

    per_class = per_class_metrics(
        cached[
            "y_true"
        ],
        predictions,
        stage_order,
    )

    row = {
        "test_subject":
            test_subject,

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

    fixed_fold_rows.append(
        row
    )

    epoch_df = (
        cached[
            "metadata"
        ].copy()
    )

    epoch_df[
        "prediction"
    ] = predictions

    fixed_epoch_frames.append(
        epoch_df
    )


fixed_fold_df = pd.DataFrame(
    fixed_fold_rows
)

fixed_epoch_df = pd.concat(
    fixed_epoch_frames,
    ignore_index=True,
)


fixed_pooled_overall = overall_metrics(
    fixed_epoch_df[
        "label"
    ],
    fixed_epoch_df[
        "prediction"
    ],
)

fixed_pooled_class = per_class_metrics(
    fixed_epoch_df[
        "label"
    ],
    fixed_epoch_df[
        "prediction"
    ],
    stage_order,
)


print("\n" + "=" * 104)
print("FIXED DEPLOYMENT PARAMETER LOSO REPORT")
print("=" * 104)

print(
    fixed_fold_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)

print(
    "\nMean Macro F1:",
    round(
        float(
            fixed_fold_df[
                "macro_f1"
            ].mean()
        ),
        4,
    )
)

print(
    "Pooled Macro F1:",
    round(
        fixed_pooled_overall[
            "macro_f1"
        ],
        4,
    )
)

print(
    "Pooled N1 F1:",
    round(
        fixed_pooled_class[
            "N1"
        ][
            "f1"
        ],
        4,
    )
)


# ============================================================
# 7. Train FINAL canonical model on ALL available data
# ============================================================

print("\n" + "=" * 104)
print("TRAINING FINAL CANONICAL MODEL")
print("=" * 104)


final_means, final_stds = (
    calculate_training_statistics(
        df_raw,
        feature_columns,
    )
)

df_normalized = (
    normalize_with_fixed_statistics(
        df_raw,
        feature_columns,
        final_means,
        final_stds,
    )
)

final_model = make_model()

training_start = time.perf_counter()

final_model.fit(
    df_normalized[
        feature_columns
    ],
    df_normalized[
        "label"
    ],
)

final_training_time_sec = (
    time.perf_counter()
    - training_start
)


(
    final_start_probabilities,
    final_transition_probabilities,
) = estimate_temporal_priors(
    df_raw,
    stage_order,
    laplace=1.0,
)


print(
    "\nFinal model training time (sec):",
    round(
        final_training_time_sec,
        4,
    )
)


# ============================================================
# 8. Save final model
# ============================================================

model_path = (
    model_dir
    / "stage43_canonical_random_forest.joblib"
)

joblib.dump(
    final_model,
    model_path,
)


# ============================================================
# 9. Save normalization statistics
# ============================================================

normalization_df = pd.DataFrame(
    {
        "feature":
            feature_columns,

        "mean":
            [
                final_means[
                    feature
                ]
                for feature in feature_columns
            ],

        "std":
            [
                final_stds[
                    feature
                ]
                for feature in feature_columns
            ],
    }
)

normalization_path = (
    model_dir
    / "stage43_normalization_statistics.csv"
)

normalization_df.to_csv(
    normalization_path,
    index=False,
)


# ============================================================
# 10. Save feature schema
# ============================================================

feature_schema_df = pd.DataFrame(
    {
        "feature_index":
            np.arange(
                len(
                    feature_columns
                )
            ),

        "feature":
            feature_columns,
    }
)

feature_schema_path = (
    model_dir
    / "stage43_feature_schema.csv"
)

feature_schema_df.to_csv(
    feature_schema_path,
    index=False,
)


# ============================================================
# 11. Save start probabilities
# ============================================================

start_probability_df = pd.DataFrame(
    {
        "stage":
            stage_order,

        "start_probability":
            final_start_probabilities,
    }
)

start_probability_path = (
    model_dir
    / "stage43_start_probabilities.csv"
)

start_probability_df.to_csv(
    start_probability_path,
    index=False,
)


# ============================================================
# 12. Save transition matrix
# ============================================================

transition_matrix_df = pd.DataFrame(
    final_transition_probabilities,
    index=stage_order,
    columns=stage_order,
)

transition_matrix_path = (
    model_dir
    / "stage43_transition_matrix.csv"
)

transition_matrix_df.to_csv(
    transition_matrix_path,
)


# ============================================================
# 13. Save fixed parameter search
# ============================================================

parameter_search_path = (
    results_dir
    / "stage43_global_parameter_search.csv"
)

parameter_search_df.to_csv(
    parameter_search_path,
    index=False,
)


fixed_fold_path = (
    results_dir
    / "stage43_fixed_parameter_loso_folds.csv"
)

fixed_fold_df.to_csv(
    fixed_fold_path,
    index=False,
)


fixed_epoch_path = (
    results_dir
    / "stage43_fixed_parameter_epoch_predictions.csv"
)

fixed_epoch_df.to_csv(
    fixed_epoch_path,
    index=False,
)


# ============================================================
# 14. Model checksum
# ============================================================

sha256 = hashlib.sha256()

with open(
    model_path,
    "rb",
) as f:

    while True:

        chunk = f.read(
            1024 * 1024
        )

        if not chunk:

            break

        sha256.update(
            chunk
        )

model_sha256 = sha256.hexdigest()


# ============================================================
# 15. Stage 40 evidence
#
# Load results when available.
# These remain the primary unbiased nested-LOSO evidence.
# ============================================================

stage40_summary_path = (
    results_dir
    / "stage40_summary.csv"
)

stage40_reference = {
    "best_mean_loso_macro_f1":
        0.6902,

    "best_pooled_macro_f1":
        0.6878,

    "best_pooled_n1_f1":
        0.3855,
}

if stage40_summary_path.exists():

    try:

        stage40_df = pd.read_csv(
            stage40_summary_path
        )

        if len(
            stage40_df
        ) > 0:

            first = stage40_df.iloc[
                0
            ]

            for key in [
                "best_mean_loso_macro_f1",
                "best_pooled_macro_f1",
                "best_pooled_n1_f1",
            ]:

                if key in stage40_df.columns:

                    stage40_reference[
                        key
                    ] = float(
                        first[
                            key
                        ]
                    )

    except Exception as exc:

        print(
            "\nWARNING:"
        )

        print(
            "Could not read Stage 40 summary:"
        )

        print(
            exc
        )


# ============================================================
# 16. Canonical manifest
# ============================================================

manifest = {
    "stage":
        43,

    "status":
        "LOCKED_CANONICAL_COMPETITION_CANDIDATE",

    "model_family":
        "RandomForestClassifier",

    "n_estimators":
        150,

    "random_state":
        42,

    "class_weight":
        "balanced",

    "feature_count":
        len(
            feature_columns
        ),

    "feature_source":
        "original Stage 40 feature set",

    "normalization":
        "training-global mean/std",

    "decoder":
        "N1-preserving Viterbi",

    "canonical_transition_weight":
        canonical_transition_weight,

    "canonical_n1_emission_boost":
        canonical_n1_emission_boost,

    "stage_order":
        stage_order,

    "training_subject_count":
        len(
            subjects
        ),

    "training_subjects":
        [
            str(
                subject
            )
            for subject in subjects
        ],

    "training_epoch_count":
        len(
            df_raw
        ),

    "final_training_time_sec":
        final_training_time_sec,

    "primary_validation_source":
        "Stage 40 nested LOSO",

    "stage40_nested_loso_mean_macro_f1":
        stage40_reference[
            "best_mean_loso_macro_f1"
        ],

    "stage40_nested_loso_pooled_macro_f1":
        stage40_reference[
            "best_pooled_macro_f1"
        ],

    "stage40_nested_loso_pooled_n1_f1":
        stage40_reference[
            "best_pooled_n1_f1"
        ],

    "fixed_parameter_loso_mean_macro_f1":
        float(
            fixed_fold_df[
                "macro_f1"
            ].mean()
        ),

    "fixed_parameter_loso_pooled_macro_f1":
        fixed_pooled_overall[
            "macro_f1"
        ],

    "fixed_parameter_loso_pooled_n1_f1":
        fixed_pooled_class[
            "N1"
        ][
            "f1"
        ],

    "model_sha256":
        model_sha256,

    "artifact_files": {
        "model":
            model_path.name,

        "normalization":
            normalization_path.name,

        "feature_schema":
            feature_schema_path.name,

        "start_probabilities":
            start_probability_path.name,

        "transition_matrix":
            transition_matrix_path.name,
    },
}


manifest_path = (
    model_dir
    / "stage43_canonical_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 17. Human-readable lock report
# ============================================================

report_lines = [
    "STAGE 43 - CANONICAL MODEL LOCK REPORT",
    "=" * 60,
    "",
    "STATUS:",
    "LOCKED_CANONICAL_COMPETITION_CANDIDATE",
    "",
    "MODEL:",
    "RandomForestClassifier",
    "n_estimators = 150",
    "class_weight = balanced",
    "random_state = 42",
    "",
    f"FEATURE COUNT = {len(feature_columns)}",
    "FEATURE SOURCE = Stage 40 original feature set",
    "",
    "DECODER:",
    "N1-preserving Viterbi",
    f"transition_weight = {canonical_transition_weight}",
    f"N1_emission_boost = {canonical_n1_emission_boost}",
    "",
    "PRIMARY UNBIASED VALIDATION:",
    "Stage 40 nested LOSO",
    (
        "Mean LOSO Macro F1 = "
        f"{stage40_reference['best_mean_loso_macro_f1']:.4f}"
    ),
    (
        "Pooled Macro F1 = "
        f"{stage40_reference['best_pooled_macro_f1']:.4f}"
    ),
    (
        "Pooled N1 F1 = "
        f"{stage40_reference['best_pooled_n1_f1']:.4f}"
    ),
    "",
    "FIXED DEPLOYMENT PARAMETER LOSO:",
    (
        "Mean Macro F1 = "
        f"{fixed_fold_df['macro_f1'].mean():.4f}"
    ),
    (
        "Pooled Macro F1 = "
        f"{fixed_pooled_overall['macro_f1']:.4f}"
    ),
    (
        "Pooled N1 F1 = "
        f"{fixed_pooled_class['N1']['f1']:.4f}"
    ),
    "",
    "IMPORTANT:",
    (
        "Use Stage 40 nested LOSO metrics for competition "
        "generalization claims."
    ),
    (
        "Use Stage 43 model artifacts for the final demo "
        "deployment pipeline."
    ),
    "",
    f"MODEL SHA256 = {model_sha256}",
]


lock_report_path = (
    model_dir
    / "stage43_lock_report.txt"
)

lock_report_path.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# ============================================================
# 18. Summary CSV
# ============================================================

summary_df = pd.DataFrame([
    {
        "canonical_feature_count":
            len(
                feature_columns
            ),

        "canonical_transition_weight":
            canonical_transition_weight,

        "canonical_n1_emission_boost":
            canonical_n1_emission_boost,

        "stage40_nested_loso_mean_macro_f1":
            stage40_reference[
                "best_mean_loso_macro_f1"
            ],

        "stage40_nested_loso_pooled_macro_f1":
            stage40_reference[
                "best_pooled_macro_f1"
            ],

        "stage40_nested_loso_pooled_n1_f1":
            stage40_reference[
                "best_pooled_n1_f1"
            ],

        "fixed_parameter_loso_mean_macro_f1":
            float(
                fixed_fold_df[
                    "macro_f1"
                ].mean()
            ),

        "fixed_parameter_loso_pooled_macro_f1":
            fixed_pooled_overall[
                "macro_f1"
            ],

        "fixed_parameter_loso_pooled_n1_f1":
            fixed_pooled_class[
                "N1"
            ][
                "f1"
            ],

        "final_model_training_time_sec":
            final_training_time_sec,

        "model_sha256":
            model_sha256,
    }
])


summary_path = (
    results_dir
    / "stage43_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# 19. Final print
# ============================================================

print("\n" + "=" * 104)
print("STAGE 43 CANONICAL LOCK SUMMARY")
print("=" * 104)

print(
    "\nCanonical feature count:",
    len(
        feature_columns
    )
)

print(
    "Canonical transition weight:",
    canonical_transition_weight
)

print(
    "Canonical N1 emission boost:",
    canonical_n1_emission_boost
)

print(
    "\nPrimary competition validation:"
)

print(
    "Stage 40 nested LOSO Mean Macro F1:",
    round(
        stage40_reference[
            "best_mean_loso_macro_f1"
        ],
        4,
    )
)

print(
    "Stage 40 pooled Macro F1:",
    round(
        stage40_reference[
            "best_pooled_macro_f1"
        ],
        4,
    )
)

print(
    "Stage 40 pooled N1 F1:",
    round(
        stage40_reference[
            "best_pooled_n1_f1"
        ],
        4,
    )
)

print(
    "\nFixed deployment parameter LOSO:"
)

print(
    "Mean Macro F1:",
    round(
        float(
            fixed_fold_df[
                "macro_f1"
            ].mean()
        ),
        4,
    )
)

print(
    "Pooled Macro F1:",
    round(
        fixed_pooled_overall[
            "macro_f1"
        ],
        4,
    )
)

print(
    "Pooled N1 F1:",
    round(
        fixed_pooled_class[
            "N1"
        ][
            "f1"
        ],
        4,
    )
)

print(
    "\nFinal model saved to:"
)

print(
    model_path
)

print(
    "\nCanonical package directory:"
)

print(
    model_dir
)

print(
    "\nModel SHA256:"
)

print(
    model_sha256
)

print(
    "\nIMPORTANT:"
)

print(
    "Use Stage 40 nested LOSO metrics for validation claims."
)

print(
    "Use Stage 43 artifacts for the final competition demo."
)

print("\n" + "=" * 104)
print("STAGE 43 COMPLETE - CANONICAL MODEL LOCKED")
print("=" * 104)
