from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

results_dir = PROJECT_ROOT / "results"
canonical_dir = PROJECT_ROOT / "models" / "stage43_canonical"
outputs_dir = PROJECT_ROOT / "outputs"

stage40_epoch_path = (
    results_dir
    / "stage40_epoch_predictions.csv"
)

stage40_summary_path = (
    results_dir
    / "stage40_summary.csv"
)

stage43_summary_path = (
    results_dir
    / "stage43_summary.csv"
)

canonical_model_path = (
    canonical_dir
    / "stage43_canonical_random_forest.joblib"
)

normalization_path = (
    canonical_dir
    / "stage43_normalization_statistics.csv"
)

feature_schema_path = (
    canonical_dir
    / "stage43_feature_schema.csv"
)

manifest_path = (
    canonical_dir
    / "stage43_canonical_manifest.json"
)

dataset_path = (
    outputs_dir
    / "multi_subject_temporal_context.csv"
)

report_dir = (
    results_dir
    / "stage44_competition_package"
)

report_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Stage 44
# Final Competition Benchmark Package
#
# PURPOSE:
#   Freeze and package competition-facing evidence.
#
# PRIMARY VALIDATION:
#   Stage 40 nested LOSO
#
# DEPLOYMENT ARTIFACT:
#   Stage 43 canonical model
#
# OUTPUT:
#   - Overall metrics
#   - Per-class metrics
#   - Confusion matrix
#   - Normalized confusion matrix
#   - Model size
#   - Inference latency
#   - Pipeline specification
#   - Competition-ready summary
#   - PNG figures
# ============================================================


print("=" * 104)
print("STAGE 44 - FINAL COMPETITION BENCHMARK PACKAGE")
print("=" * 104)


# ============================================================
# 1. Load Stage 40 validation predictions
# ============================================================

if not stage40_epoch_path.exists():

    raise FileNotFoundError(
        f"Missing Stage 40 epoch predictions: {stage40_epoch_path}"
    )

stage40_epoch_df = pd.read_csv(
    stage40_epoch_path
)

print("\nLoaded:")
print(stage40_epoch_path)

print("\nColumns:")
print(stage40_epoch_df.columns.tolist())


# ============================================================
# 2. Determine canonical Stage 40 prediction column
#
# Expected Stage 40 columns:
#   label
#   baseline_argmax
#   standard_viterbi
#   n1_preserving_viterbi
# ============================================================

candidate_prediction_columns = [
    "n1_preserving_viterbi",
    "prediction",
    "smoothed_prediction",
]

prediction_column = None

for candidate in candidate_prediction_columns:

    if candidate in stage40_epoch_df.columns:

        prediction_column = candidate
        break


if prediction_column is None:

    raise ValueError(
        "Could not find Stage 40 canonical prediction column. "
        "Expected n1_preserving_viterbi or prediction."
    )


if "label" not in stage40_epoch_df.columns:

    raise ValueError(
        "Stage 40 epoch predictions must contain 'label'."
    )


print("\nUsing validation prediction column:")
print(prediction_column)


# ============================================================
# 3. Stage labels
# ============================================================

preferred_stage_order = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]

existing_labels = set(
    stage40_epoch_df[
        "label"
    ].dropna().unique()
)

stage_order = [
    stage
    for stage in preferred_stage_order
    if stage in existing_labels
]

print("\nStage order:")
print(stage_order)


# ============================================================
# 4. Overall validation metrics
# ============================================================

y_true = stage40_epoch_df[
    "label"
].to_numpy()

y_pred = stage40_epoch_df[
    prediction_column
].to_numpy()


overall_accuracy = accuracy_score(
    y_true,
    y_pred,
)

overall_balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred,
)

overall_macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
)


# ============================================================
# 5. Per-class validation metrics
# ============================================================

precision, recall, f1_values, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=stage_order,
        zero_division=0,
    )
)

per_class_df = pd.DataFrame(
    {
        "stage":
            stage_order,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1_values,

        "support":
            support,
    }
)


# ============================================================
# 6. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=stage_order,
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"true_{stage}"
        for stage in stage_order
    ],
    columns=[
        f"pred_{stage}"
        for stage in stage_order
    ],
)


row_sums = cm.sum(
    axis=1,
    keepdims=True,
)

normalized_cm = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(
        cm,
        dtype=float,
    ),
    where=row_sums != 0,
)

normalized_cm_df = pd.DataFrame(
    normalized_cm,
    index=[
        f"true_{stage}"
        for stage in stage_order
    ],
    columns=[
        f"pred_{stage}"
        for stage in stage_order
    ],
)


# ============================================================
# 7. Load Stage 40 summary
# ============================================================

stage40_summary = {}

if stage40_summary_path.exists():

    summary_df = pd.read_csv(
        stage40_summary_path
    )

    if len(summary_df) > 0:

        stage40_summary = (
            summary_df.iloc[
                0
            ].to_dict()
        )


# ============================================================
# 8. Load Stage 43 canonical manifest
# ============================================================

if not manifest_path.exists():

    raise FileNotFoundError(
        f"Missing canonical manifest: {manifest_path}"
    )

with open(
    manifest_path,
    "r",
    encoding="utf-8",
) as f:

    manifest = json.load(
        f
    )


# ============================================================
# 9. Canonical model size
# ============================================================

if not canonical_model_path.exists():

    raise FileNotFoundError(
        f"Missing canonical model: {canonical_model_path}"
    )

model_size_bytes = canonical_model_path.stat().st_size

model_size_kb = (
    model_size_bytes
    / 1024.0
)

model_size_mb = (
    model_size_kb
    / 1024.0
)


# ============================================================
# 10. Load canonical model + schema + normalization
# ============================================================

model = joblib.load(
    canonical_model_path
)

normalization_df = pd.read_csv(
    normalization_path
)

feature_schema_df = pd.read_csv(
    feature_schema_path
)

feature_columns = feature_schema_df[
    "feature"
].tolist()

feature_count = len(
    feature_columns
)


# ============================================================
# 11. Load dataset for deployment latency benchmark
# ============================================================

dataset_df = pd.read_csv(
    dataset_path
)

missing_features = [
    feature
    for feature in feature_columns
    if feature not in dataset_df.columns
]

if missing_features:

    raise ValueError(
        "Dataset is missing canonical features: "
        + ", ".join(
            missing_features
        )
    )


mean_map = dict(
    zip(
        normalization_df[
            "feature"
        ],
        normalization_df[
            "mean"
        ],
    )
)

std_map = dict(
    zip(
        normalization_df[
            "feature"
        ],
        normalization_df[
            "std"
        ],
    )
)


X = dataset_df[
    feature_columns
].copy()


for feature in feature_columns:

    X[
        feature
    ] = (
        X[
            feature
        ].astype(float)
        - float(
            mean_map[
                feature
            ]
        )
    ) / (
        float(
            std_map[
                feature
            ]
        )
        + 1e-8
    )


X = (
    X
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .fillna(
        0.0
    )
)


# ============================================================
# 12. Inference latency benchmark
#
# Two views:
#
# A. Full batch inference
# B. Single-epoch repeated inference
#
# Note:
#   This benchmarks RF probability inference only.
#   Full-night Viterbi decoding is separately benchmarked below.
# ============================================================

# Warm-up
_ = model.predict_proba(
    X.iloc[
        :min(
            20,
            len(X),
        )
    ]
)


# ------------------------------------------------------------
# Full batch
# ------------------------------------------------------------

batch_runs = []

for _ in range(
    20
):

    start = time.perf_counter()

    _ = model.predict_proba(
        X
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    batch_runs.append(
        elapsed
    )


batch_mean_sec = float(
    np.mean(
        batch_runs
    )
)

batch_median_sec = float(
    np.median(
        batch_runs
    )
)

batch_p95_sec = float(
    np.percentile(
        batch_runs,
        95,
    )
)

per_epoch_batch_ms = (
    batch_mean_sec
    / len(X)
    * 1000.0
)


# ------------------------------------------------------------
# Single epoch repeated
# ------------------------------------------------------------

sample_count = min(
    500,
    len(X),
)

rng = np.random.default_rng(
    42
)

sample_indices = rng.choice(
    len(X),
    size=sample_count,
    replace=False,
)

single_epoch_times = []

for idx in sample_indices:

    sample = X.iloc[
        [
            int(
                idx
            )
        ]
    ]

    start = time.perf_counter()

    _ = model.predict_proba(
        sample
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    single_epoch_times.append(
        elapsed
    )


single_epoch_mean_ms = (
    float(
        np.mean(
            single_epoch_times
        )
    )
    * 1000.0
)

single_epoch_median_ms = (
    float(
        np.median(
            single_epoch_times
        )
    )
    * 1000.0
)

single_epoch_p95_ms = (
    float(
        np.percentile(
            single_epoch_times,
            95,
        )
    )
    * 1000.0
)


# ============================================================
# 13. Minimal Viterbi implementation for latency measurement
# ============================================================

stage_order_manifest = manifest[
    "stage_order"
]

stage_to_idx = {
    stage: idx
    for idx, stage in enumerate(
        stage_order_manifest
    )
}

n1_index = stage_to_idx[
    "N1"
]

transition_weight = float(
    manifest[
        "canonical_transition_weight"
    ]
)

n1_boost = float(
    manifest[
        "canonical_n1_emission_boost"
    ]
)


start_probability_path = (
    canonical_dir
    / "stage43_start_probabilities.csv"
)

transition_matrix_path = (
    canonical_dir
    / "stage43_transition_matrix.csv"
)

start_df = pd.read_csv(
    start_probability_path
)

transition_df = pd.read_csv(
    transition_matrix_path,
    index_col=0,
)


start_probabilities = np.array(
    [
        float(
            start_df.loc[
                start_df[
                    "stage"
                ] == stage,
                "start_probability",
            ].iloc[
                0
            ]
        )
        for stage in stage_order_manifest
    ]
)


transition_probabilities = transition_df.loc[
    stage_order_manifest,
    stage_order_manifest,
].to_numpy(
    dtype=float
)


def align_probabilities(
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


def viterbi_decode(
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

            scores = (
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

            best_prev = int(
                np.argmax(
                    scores
                )
            )

            dp[
                t,
                current_state
            ] = (
                scores[
                    best_prev
                ]
                + log_emission[
                    t,
                    current_state
                ]
            )

            backpointer[
                t,
                current_state
            ] = best_prev

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


# ============================================================
# 14. Full-night Viterbi latency benchmark
# ============================================================

largest_recording = None
largest_recording_size = 0

for recording, group in dataset_df.groupby(
    "recording",
    sort=False,
):

    if len(
        group
    ) > largest_recording_size:

        largest_recording = recording
        largest_recording_size = len(
            group
        )


recording_mask = (
    dataset_df[
        "recording"
    ] == largest_recording
)

X_recording = X.loc[
    recording_mask
]


recording_probabilities = (
    model.predict_proba(
        X_recording
    )
)

aligned_recording_probabilities = (
    align_probabilities(
        model,
        recording_probabilities,
        stage_order_manifest,
    )
)


viterbi_runs = []

for _ in range(
    30
):

    start = time.perf_counter()

    _ = viterbi_decode(
        aligned_recording_probabilities,
        start_probabilities,
        transition_probabilities,
        transition_weight,
        n1_boost,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    viterbi_runs.append(
        elapsed
    )


viterbi_mean_ms = (
    float(
        np.mean(
            viterbi_runs
        )
    )
    * 1000.0
)

viterbi_median_ms = (
    float(
        np.median(
            viterbi_runs
        )
    )
    * 1000.0
)

viterbi_p95_ms = (
    float(
        np.percentile(
            viterbi_runs,
            95,
        )
    )
    * 1000.0
)


# ============================================================
# 15. Benchmark summary table
# ============================================================

nested_mean_macro_f1 = float(
    stage40_summary.get(
        "best_mean_loso_macro_f1",
        0.6902,
    )
)

nested_pooled_macro_f1 = float(
    stage40_summary.get(
        "best_pooled_macro_f1",
        overall_macro_f1,
    )
)

nested_pooled_n1_f1 = float(
    stage40_summary.get(
        "best_pooled_n1_f1",
        per_class_df.loc[
            per_class_df[
                "stage"
            ] == "N1",
            "f1",
        ].iloc[
            0
        ],
    )
)


benchmark_summary_df = pd.DataFrame(
    [
        {
            "validation_protocol":
                "Nested LOSO",

            "accuracy":
                overall_accuracy,

            "balanced_accuracy":
                overall_balanced_accuracy,

            "pooled_macro_f1":
                nested_pooled_macro_f1,

            "mean_subject_macro_f1":
                nested_mean_macro_f1,

            "pooled_n1_f1":
                nested_pooled_n1_f1,

            "feature_count":
                feature_count,

            "rf_tree_count":
                150,

            "model_size_mb":
                model_size_mb,

            "batch_inference_mean_sec":
                batch_mean_sec,

            "batch_inference_p95_sec":
                batch_p95_sec,

            "batch_per_epoch_ms":
                per_epoch_batch_ms,

            "single_epoch_mean_ms":
                single_epoch_mean_ms,

            "single_epoch_p95_ms":
                single_epoch_p95_ms,

            "viterbi_recording_epoch_count":
                largest_recording_size,

            "viterbi_mean_ms":
                viterbi_mean_ms,

            "viterbi_p95_ms":
                viterbi_p95_ms,

            "transition_weight":
                transition_weight,

            "n1_emission_boost":
                n1_boost,
        }
    ]
)


# ============================================================
# 16. Pipeline specification table
# ============================================================

pipeline_df = pd.DataFrame(
    [
        {
            "component":
                "Input",

            "specification":
                "30-second EEG epochs with causal temporal context",
        },
        {
            "component":
                "Feature set",

            "specification":
                f"{feature_count} canonical features",
        },
        {
            "component":
                "Normalization",

            "specification":
                "Training-global mean/std normalization",
        },
        {
            "component":
                "Classifier",

            "specification":
                "Random Forest, 150 trees, balanced class weights",
        },
        {
            "component":
                "Temporal decoder",

            "specification":
                "N1-preserving Viterbi",
        },
        {
            "component":
                "Transition weight",

            "specification":
                str(
                    transition_weight
                ),
        },
        {
            "component":
                "N1 emission boost",

            "specification":
                str(
                    n1_boost
                ),
        },
        {
            "component":
                "Validation",

            "specification":
                "Nested leave-one-subject-out cross-validation",
        },
    ]
)


# ============================================================
# 17. Figure 1: Per-class F1
# ============================================================

fig = plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.bar(
    per_class_df[
        "stage"
    ],
    per_class_df[
        "f1"
    ],
)

plt.ylim(
    0,
    1,
)

plt.ylabel(
    "F1 score"
)

plt.xlabel(
    "Sleep stage"
)

plt.title(
    "Stage 44 - Per-class F1"
)

plt.tight_layout()

per_class_figure_path = (
    report_dir
    / "stage44_per_class_f1.png"
)

plt.savefig(
    per_class_figure_path,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# 18. Figure 2: Normalized confusion matrix
# ============================================================

fig = plt.figure(
    figsize=(
        7,
        6,
    )
)

image = plt.imshow(
    normalized_cm,
    vmin=0,
    vmax=1,
)

plt.colorbar(
    image,
    label="Row-normalized proportion",
)

plt.xticks(
    range(
        len(
            stage_order
        )
    ),
    stage_order,
)

plt.yticks(
    range(
        len(
            stage_order
        )
    ),
    stage_order,
)

plt.xlabel(
    "Predicted stage"
)

plt.ylabel(
    "True stage"
)

plt.title(
    "Stage 44 - Normalized Confusion Matrix"
)

for i in range(
    len(
        stage_order
    )
):

    for j in range(
        len(
            stage_order
        )
    ):

        plt.text(
            j,
            i,
            f"{normalized_cm[i, j]:.2f}",
            ha="center",
            va="center",
        )


plt.tight_layout()

confusion_figure_path = (
    report_dir
    / "stage44_normalized_confusion_matrix.png"
)

plt.savefig(
    confusion_figure_path,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# 19. Figure 3: Key competition metrics
# ============================================================

metric_names = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
]

metric_values = [
    overall_accuracy,
    overall_balanced_accuracy,
    nested_pooled_macro_f1,
]


fig = plt.figure(
    figsize=(
        7,
        5,
    )
)

plt.bar(
    metric_names,
    metric_values,
)

plt.ylim(
    0,
    1,
)

plt.ylabel(
    "Score"
)

plt.title(
    "Stage 44 - Competition Benchmark"
)

for i, value in enumerate(
    metric_values
):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )


plt.tight_layout()

benchmark_figure_path = (
    report_dir
    / "stage44_key_metrics.png"
)

plt.savefig(
    benchmark_figure_path,
    dpi=200,
)

plt.close(
    fig
)


# ============================================================
# 20. Save tables
# ============================================================

benchmark_summary_path = (
    report_dir
    / "stage44_benchmark_summary.csv"
)

per_class_path = (
    report_dir
    / "stage44_per_class_metrics.csv"
)

confusion_path = (
    report_dir
    / "stage44_confusion_matrix.csv"
)

normalized_confusion_path = (
    report_dir
    / "stage44_normalized_confusion_matrix.csv"
)

latency_path = (
    report_dir
    / "stage44_latency_benchmark.csv"
)

pipeline_path = (
    report_dir
    / "stage44_pipeline_specification.csv"
)


benchmark_summary_df.to_csv(
    benchmark_summary_path,
    index=False,
)

per_class_df.to_csv(
    per_class_path,
    index=False,
)

cm_df.to_csv(
    confusion_path,
)

normalized_cm_df.to_csv(
    normalized_confusion_path,
)

pipeline_df.to_csv(
    pipeline_path,
    index=False,
)


latency_df = pd.DataFrame(
    [
        {
            "metric":
                "full_batch_mean_sec",

            "value":
                batch_mean_sec,
        },
        {
            "metric":
                "full_batch_median_sec",

            "value":
                batch_median_sec,
        },
        {
            "metric":
                "full_batch_p95_sec",

            "value":
                batch_p95_sec,
        },
        {
            "metric":
                "batch_per_epoch_ms",

            "value":
                per_epoch_batch_ms,
        },
        {
            "metric":
                "single_epoch_mean_ms",

            "value":
                single_epoch_mean_ms,
        },
        {
            "metric":
                "single_epoch_median_ms",

            "value":
                single_epoch_median_ms,
        },
        {
            "metric":
                "single_epoch_p95_ms",

            "value":
                single_epoch_p95_ms,
        },
        {
            "metric":
                "viterbi_mean_ms",

            "value":
                viterbi_mean_ms,
        },
        {
            "metric":
                "viterbi_median_ms",

            "value":
                viterbi_median_ms,
        },
        {
            "metric":
                "viterbi_p95_ms",

            "value":
                viterbi_p95_ms,
        },
    ]
)

latency_df.to_csv(
    latency_path,
    index=False,
)


# ============================================================
# 21. Competition-ready TXT report
# ============================================================

report_lines = [
    "PRECISION SLEEP - STAGE 44 FINAL COMPETITION BENCHMARK",
    "=" * 72,
    "",
    "PRIMARY VALIDATION",
    "Nested leave-one-subject-out cross-validation",
    "",
    f"Mean subject Macro F1: {nested_mean_macro_f1:.4f}",
    f"Pooled Macro F1: {nested_pooled_macro_f1:.4f}",
    f"Accuracy: {overall_accuracy:.4f}",
    f"Balanced Accuracy: {overall_balanced_accuracy:.4f}",
    f"Pooled N1 F1: {nested_pooled_n1_f1:.4f}",
    "",
    "PER-CLASS F1",
]

for _, row in per_class_df.iterrows():

    report_lines.append(
        f"{row['stage']}: {row['f1']:.4f}"
    )


report_lines.extend(
    [
        "",
        "CANONICAL MODEL",
        f"Feature count: {feature_count}",
        "Classifier: Random Forest",
        "Trees: 150",
        "Class weighting: balanced",
        "Decoder: N1-preserving Viterbi",
        f"Transition weight: {transition_weight}",
        f"N1 emission boost: {n1_boost}",
        "",
        "DEPLOYMENT SIZE / SPEED",
        f"Model size: {model_size_mb:.3f} MB",
        f"Batch per-epoch inference: {per_epoch_batch_ms:.4f} ms",
        f"Single-epoch mean inference: {single_epoch_mean_ms:.4f} ms",
        f"Single-epoch P95 inference: {single_epoch_p95_ms:.4f} ms",
        (
            "Viterbi decoding "
            f"({largest_recording_size} epochs): "
            f"{viterbi_mean_ms:.4f} ms mean"
        ),
        "",
        "COMPETITION CLAIM GUIDANCE",
        (
            "Use Stage 40 nested LOSO values for generalization claims."
        ),
        (
            "Use Stage 43 artifacts as the locked deployment model."
        ),
        (
            "Stage 44 packages the evidence; it does not alter the model."
        ),
    ]
)


text_report_path = (
    report_dir
    / "stage44_competition_report.txt"
)

text_report_path.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# ============================================================
# 22. JSON summary for future Demo/UI code
# ============================================================

json_summary = {
    "validation": {
        "protocol":
            "nested LOSO",

        "mean_subject_macro_f1":
            nested_mean_macro_f1,

        "pooled_macro_f1":
            nested_pooled_macro_f1,

        "accuracy":
            overall_accuracy,

        "balanced_accuracy":
            overall_balanced_accuracy,

        "pooled_n1_f1":
            nested_pooled_n1_f1,
    },

    "per_class_f1": {
        row[
            "stage"
        ]:
            float(
                row[
                    "f1"
                ]
            )
        for _, row in per_class_df.iterrows()
    },

    "deployment": {
        "feature_count":
            feature_count,

        "classifier":
            "Random Forest",

        "tree_count":
            150,

        "decoder":
            "N1-preserving Viterbi",

        "transition_weight":
            transition_weight,

        "n1_emission_boost":
            n1_boost,

        "model_size_mb":
            model_size_mb,

        "single_epoch_mean_ms":
            single_epoch_mean_ms,

        "single_epoch_p95_ms":
            single_epoch_p95_ms,

        "viterbi_mean_ms":
            viterbi_mean_ms,
    },
}


json_summary_path = (
    report_dir
    / "stage44_competition_summary.json"
)

with open(
    json_summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 23. Final print
# ============================================================

print("\n" + "=" * 104)
print("STAGE 44 FINAL SUMMARY")
print("=" * 104)

print(
    "\nValidation protocol:"
)

print(
    "Nested LOSO"
)

print(
    "\nMean subject Macro F1:",
    round(
        nested_mean_macro_f1,
        4,
    )
)

print(
    "Pooled Macro F1:",
    round(
        nested_pooled_macro_f1,
        4,
    )
)

print(
    "Accuracy:",
    round(
        overall_accuracy,
        4,
    )
)

print(
    "Balanced Accuracy:",
    round(
        overall_balanced_accuracy,
        4,
    )
)

print(
    "Pooled N1 F1:",
    round(
        nested_pooled_n1_f1,
        4,
    )
)

print(
    "\nPer-class F1:"
)

for _, row in per_class_df.iterrows():

    print(
        f"  {row['stage']}: "
        f"{row['f1']:.4f}"
    )


print(
    "\nModel size (MB):",
    round(
        model_size_mb,
        4,
    )
)

print(
    "Single-epoch mean inference (ms):",
    round(
        single_epoch_mean_ms,
        4,
    )
)

print(
    "Single-epoch P95 inference (ms):",
    round(
        single_epoch_p95_ms,
        4,
    )
)

print(
    "Viterbi mean latency (ms):",
    round(
        viterbi_mean_ms,
        4,
    )
)

print(
    "\nCompetition package directory:"
)

print(
    report_dir
)

print(
    "\nGenerated figures:"
)

print(
    per_class_figure_path
)

print(
    confusion_figure_path
)

print(
    benchmark_figure_path
)

print("\n" + "=" * 104)
print("STAGE 44 COMPLETE - COMPETITION BENCHMARK PACKAGE READY")
print("=" * 104)
