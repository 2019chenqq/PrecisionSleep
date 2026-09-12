from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

results_dir = (
    PROJECT_ROOT
    / "results"
)

input_path = (
    results_dir
    / "stage33_loso_epoch_predictions.csv"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Stage 34
# Cross-subject / Per-class Error Analysis
#
# Input:
#   stage33_loso_epoch_predictions.csv
#
# Goals:
#   1. Identify which sleep stages have the weakest F1.
#   2. Identify which subject-stage combinations fail most.
#   3. Quantify the most common confusion pairs.
#   4. Compare raw vs smoothed predictions.
#   5. Find whether low LOSO performance is concentrated
#      in one class (e.g. N1) or distributed across classes.
# ============================================================


print("=" * 82)
print("STAGE 34 - CROSS-SUBJECT ERROR ANALYSIS")
print("=" * 82)


# ============================================================
# 1. Load Stage 33 predictions
# ============================================================

if not input_path.exists():

    raise FileNotFoundError(
        "Stage 33 prediction file was not found:\n"
        f"{input_path}\n\n"
        "Run Stage 33 first."
    )


df = pd.read_csv(
    input_path
)

required_columns = [
    "subject",
    "recording",
    "epoch",
    "label",
    "raw_prediction",
    "smoothed_prediction",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Stage 33 prediction file is missing columns:\n"
        + ", ".join(
            missing_columns
        )
    )


print("\nLoaded:")
print(input_path)

print("\nTotal epochs:")
print(len(df))

subjects = sorted(
    df[
        "subject"
    ].unique()
)

print("\nSubjects:")
print(subjects)


# ============================================================
# 2. Canonical stage order
# ============================================================

preferred_stage_order = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]

existing_labels = set(
    df[
        "label"
    ].astype(str).unique()
)

stage_order = [
    stage
    for stage in preferred_stage_order
    if stage in existing_labels
]

extra_labels = sorted(
    existing_labels
    - set(stage_order)
)

stage_order.extend(
    extra_labels
)

print("\nStage order:")
print(stage_order)


# ============================================================
# 3. Overall metric helper
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


# ============================================================
# 4. Per-class metric helper
# ============================================================

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

    rows = []

    for (
        stage,
        stage_precision,
        stage_recall,
        stage_f1,
        stage_support,
    ) in zip(
        labels,
        precision,
        recall,
        f1,
        support,
    ):

        rows.append(
            {
                "stage":
                    stage,

                "precision":
                    float(
                        stage_precision
                    ),

                "recall":
                    float(
                        stage_recall
                    ),

                "f1":
                    float(
                        stage_f1
                    ),

                "support":
                    int(
                        stage_support
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. Overall raw vs smoothed metrics
# ============================================================

y_true = df[
    "label"
]

y_raw = df[
    "raw_prediction"
]

y_smoothed = df[
    "smoothed_prediction"
]


raw_overall = overall_metrics(
    y_true,
    y_raw,
)

smoothed_overall = overall_metrics(
    y_true,
    y_smoothed,
)


print("\n" + "=" * 82)
print("OVERALL PERFORMANCE")
print("=" * 82)

print("\nRaw")

print(
    "Accuracy:",
    round(
        raw_overall[
            "accuracy"
        ],
        4,
    )
)

print(
    "Balanced Accuracy:",
    round(
        raw_overall[
            "balanced_accuracy"
        ],
        4,
    )
)

print(
    "Macro F1:",
    round(
        raw_overall[
            "macro_f1"
        ],
        4,
    )
)


print("\nSmoothed")

print(
    "Accuracy:",
    round(
        smoothed_overall[
            "accuracy"
        ],
        4,
    )
)

print(
    "Balanced Accuracy:",
    round(
        smoothed_overall[
            "balanced_accuracy"
        ],
        4,
    )
)

print(
    "Macro F1:",
    round(
        smoothed_overall[
            "macro_f1"
        ],
        4,
    )
)


# ============================================================
# 6. Overall per-class metrics
# ============================================================

raw_class_df = per_class_metrics(
    y_true,
    y_raw,
    stage_order,
)

raw_class_df[
    "prediction_type"
] = "raw"


smoothed_class_df = per_class_metrics(
    y_true,
    y_smoothed,
    stage_order,
)

smoothed_class_df[
    "prediction_type"
] = "smoothed"


overall_class_df = pd.concat(
    [
        raw_class_df,
        smoothed_class_df,
    ],
    ignore_index=True,
)


print("\n" + "=" * 82)
print("SMOOTHED PER-CLASS PERFORMANCE")
print("=" * 82)

print(
    smoothed_class_df[
        [
            "stage",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 7. Weakest class
# ============================================================

weakest_class_row = (
    smoothed_class_df
    .sort_values(
        by="f1",
        ascending=True,
    )
    .iloc[0]
)

weakest_stage = str(
    weakest_class_row[
        "stage"
    ]
)

weakest_stage_f1 = float(
    weakest_class_row[
        "f1"
    ]
)


# ============================================================
# 8. Per-subject overall metrics
# ============================================================

subject_rows = []

for subject in subjects:

    subject_df = df[
        df[
            "subject"
        ] == subject
    ]

    subject_raw = overall_metrics(
        subject_df[
            "label"
        ],
        subject_df[
            "raw_prediction"
        ],
    )

    subject_smoothed = overall_metrics(
        subject_df[
            "label"
        ],
        subject_df[
            "smoothed_prediction"
        ],
    )

    subject_rows.append(
        {
            "subject":
                subject,

            "epoch_count":
                len(
                    subject_df
                ),

            "raw_accuracy":
                subject_raw[
                    "accuracy"
                ],

            "raw_balanced_accuracy":
                subject_raw[
                    "balanced_accuracy"
                ],

            "raw_macro_f1":
                subject_raw[
                    "macro_f1"
                ],

            "smoothed_accuracy":
                subject_smoothed[
                    "accuracy"
                ],

            "smoothed_balanced_accuracy":
                subject_smoothed[
                    "balanced_accuracy"
                ],

            "smoothed_macro_f1":
                subject_smoothed[
                    "macro_f1"
                ],

            "macro_f1_delta":
                subject_smoothed[
                    "macro_f1"
                ]
                - subject_raw[
                    "macro_f1"
                ],
        }
    )


subject_overall_df = pd.DataFrame(
    subject_rows
)

subject_overall_df = (
    subject_overall_df
    .sort_values(
        by="smoothed_macro_f1",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 82)
print("SUBJECT PERFORMANCE")
print("=" * 82)

print(
    subject_overall_df[
        [
            "subject",
            "smoothed_accuracy",
            "smoothed_balanced_accuracy",
            "smoothed_macro_f1",
            "macro_f1_delta",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 9. Per-subject, per-stage metrics
# ============================================================

subject_stage_rows = []

for subject in subjects:

    subject_df = df[
        df[
            "subject"
        ] == subject
    ]

    metrics_df = per_class_metrics(
        subject_df[
            "label"
        ],
        subject_df[
            "smoothed_prediction"
        ],
        stage_order,
    )

    metrics_df[
        "subject"
    ] = subject

    subject_stage_rows.append(
        metrics_df
    )


subject_stage_df = pd.concat(
    subject_stage_rows,
    ignore_index=True,
)


# ============================================================
# 10. Subject x stage F1 pivot
# ============================================================

subject_stage_f1_pivot = (
    subject_stage_df
    .pivot(
        index="subject",
        columns="stage",
        values="f1",
    )
)

subject_stage_f1_pivot = (
    subject_stage_f1_pivot
    .reindex(
        columns=stage_order
    )
)


print("\n" + "=" * 82)
print("SUBJECT x STAGE F1")
print("=" * 82)

print(
    subject_stage_f1_pivot.to_string(
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 11. Per-subject weakest stage
# ============================================================

weakest_subject_stage_rows = []

for subject in subjects:

    subset = (
        subject_stage_df[
            subject_stage_df[
                "subject"
            ] == subject
        ]
        .sort_values(
            by="f1",
            ascending=True,
        )
    )

    weakest = subset.iloc[
        0
    ]

    weakest_subject_stage_rows.append(
        {
            "subject":
                subject,

            "weakest_stage":
                weakest[
                    "stage"
                ],

            "weakest_stage_f1":
                weakest[
                    "f1"
                ],

            "weakest_stage_precision":
                weakest[
                    "precision"
                ],

            "weakest_stage_recall":
                weakest[
                    "recall"
                ],

            "weakest_stage_support":
                weakest[
                    "support"
                ],
        }
    )


weakest_subject_stage_df = pd.DataFrame(
    weakest_subject_stage_rows
)


# ============================================================
# 12. Overall confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_smoothed,
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


# ============================================================
# 13. Row-normalized confusion matrix
# ============================================================

row_sums = cm.sum(
    axis=1,
    keepdims=True,
)

cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(
        cm,
        dtype=float,
    ),
    where=row_sums != 0,
)

cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=[
        f"true_{stage}"
        for stage in stage_order
    ],
    columns=[
        f"pred_{stage}"
        for stage in stage_order
    ],
)


print("\n" + "=" * 82)
print("NORMALIZED CONFUSION MATRIX")
print("=" * 82)

print(
    cm_normalized_df.to_string(
        float_format=lambda x: (
            f"{x:.3f}"
        ),
    )
)


# ============================================================
# 14. Error pair analysis
# ============================================================

error_df = df[
    df[
        "label"
    ] != df[
        "smoothed_prediction"
    ]
].copy()


if len(error_df) > 0:

    error_pair_df = (
        error_df
        .groupby(
            [
                "label",
                "smoothed_prediction",
            ]
        )
        .size()
        .reset_index(
            name="error_count"
        )
        .rename(
            columns={
                "label":
                    "true_stage",

                "smoothed_prediction":
                    "predicted_stage",
            }
        )
        .sort_values(
            by="error_count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    total_errors = int(
        error_pair_df[
            "error_count"
        ].sum()
    )

    error_pair_df[
        "percent_of_all_errors"
    ] = (
        error_pair_df[
            "error_count"
        ]
        / total_errors
        * 100
    )

else:

    error_pair_df = pd.DataFrame(
        columns=[
            "true_stage",
            "predicted_stage",
            "error_count",
            "percent_of_all_errors",
        ]
    )


print("\n" + "=" * 82)
print("TOP CONFUSION PAIRS")
print("=" * 82)

print(
    error_pair_df
    .head(
        10
    )
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.2f}"
        ),
    )
)


# ============================================================
# 15. Per-subject confusion pairs
# ============================================================

subject_confusion_rows = []

for subject in subjects:

    subject_df = df[
        df[
            "subject"
        ] == subject
    ]

    subject_errors = subject_df[
        subject_df[
            "label"
        ] != subject_df[
            "smoothed_prediction"
        ]
    ]

    if len(
        subject_errors
    ) == 0:

        continue

    counts = (
        subject_errors
        .groupby(
            [
                "label",
                "smoothed_prediction",
            ]
        )
        .size()
        .reset_index(
            name="error_count"
        )
        .sort_values(
            by="error_count",
            ascending=False,
        )
    )

    subject_total_errors = int(
        counts[
            "error_count"
        ].sum()
    )

    for _, row in counts.iterrows():

        subject_confusion_rows.append(
            {
                "subject":
                    subject,

                "true_stage":
                    row[
                        "label"
                    ],

                "predicted_stage":
                    row[
                        "smoothed_prediction"
                    ],

                "error_count":
                    int(
                        row[
                            "error_count"
                        ]
                    ),

                "percent_of_subject_errors":
                    float(
                        row[
                            "error_count"
                        ]
                        / subject_total_errors
                        * 100
                    ),
            }
        )


subject_confusion_df = pd.DataFrame(
    subject_confusion_rows
)


# ============================================================
# 16. Stage support distribution
# ============================================================

support_df = (
    df[
        "label"
    ]
    .value_counts()
    .reindex(
        stage_order,
        fill_value=0,
    )
    .rename_axis(
        "stage"
    )
    .reset_index(
        name="epoch_count"
    )
)

support_df[
    "percent_of_dataset"
] = (
    support_df[
        "epoch_count"
    ]
    / len(df)
    * 100
)


print("\n" + "=" * 82)
print("CLASS DISTRIBUTION")
print("=" * 82)

print(
    support_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.2f}"
        ),
    )
)


# ============================================================
# 17. Find worst subject and inspect class-level failures
# ============================================================

worst_subject_row = (
    subject_overall_df
    .sort_values(
        by="smoothed_macro_f1",
        ascending=True,
    )
    .iloc[0]
)

worst_subject = str(
    worst_subject_row[
        "subject"
    ]
)

worst_subject_f1 = float(
    worst_subject_row[
        "smoothed_macro_f1"
    ]
)

worst_subject_stage_df = (
    subject_stage_df[
        subject_stage_df[
            "subject"
        ] == worst_subject
    ]
    .sort_values(
        by="f1",
        ascending=True,
    )
)


print("\n" + "=" * 82)
print("WORST SUBJECT BREAKDOWN")
print("=" * 82)

print(
    "Worst subject:",
    worst_subject
)

print(
    "Smoothed Macro F1:",
    round(
        worst_subject_f1,
        4,
    )
)

print(
    "\nPer-stage metrics:"
)

print(
    worst_subject_stage_df[
        [
            "stage",
            "precision",
            "recall",
            "f1",
            "support",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# 18. Plot confusion matrix
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

image = ax.imshow(
    cm_normalized,
)

ax.set_xticks(
    np.arange(
        len(stage_order)
    )
)

ax.set_yticks(
    np.arange(
        len(stage_order)
    )
)

ax.set_xticklabels(
    stage_order
)

ax.set_yticklabels(
    stage_order
)

ax.set_xlabel(
    "Predicted stage"
)

ax.set_ylabel(
    "True stage"
)

ax.set_title(
    "Stage 34 - Normalized Confusion Matrix"
)

for i in range(
    len(stage_order)
):

    for j in range(
        len(stage_order)
    ):

        ax.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            ha="center",
            va="center",
        )

fig.colorbar(
    image,
    ax=ax,
)

fig.tight_layout()

confusion_plot_path = (
    results_dir
    / "stage34_confusion_matrix.png"
)

plt.savefig(
    confusion_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 19. Plot per-class F1
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.bar(
    smoothed_class_df[
        "stage"
    ],
    smoothed_class_df[
        "f1"
    ],
)

ax.set_ylim(
    0,
    1,
)

ax.set_xlabel(
    "Sleep stage"
)

ax.set_ylabel(
    "F1 score"
)

ax.set_title(
    "Stage 34 - Smoothed Per-Class F1"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

for i, value in enumerate(
    smoothed_class_df[
        "f1"
    ]
):

    ax.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

fig.tight_layout()

class_f1_plot_path = (
    results_dir
    / "stage34_per_class_f1.png"
)

plt.savefig(
    class_f1_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. Plot subject x stage F1 heatmap
# ============================================================

heatmap_values = (
    subject_stage_f1_pivot
    .to_numpy()
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

image = ax.imshow(
    heatmap_values,
    aspect="auto",
    vmin=0,
    vmax=1,
)

ax.set_xticks(
    np.arange(
        len(
            subject_stage_f1_pivot.columns
        )
    )
)

ax.set_xticklabels(
    subject_stage_f1_pivot.columns
)

ax.set_yticks(
    np.arange(
        len(
            subject_stage_f1_pivot.index
        )
    )
)

ax.set_yticklabels(
    subject_stage_f1_pivot.index
)

ax.set_xlabel(
    "Sleep stage"
)

ax.set_ylabel(
    "Subject"
)

ax.set_title(
    "Stage 34 - Subject x Stage F1"
)

for i in range(
    heatmap_values.shape[0]
):

    for j in range(
        heatmap_values.shape[1]
    ):

        value = heatmap_values[
            i,
            j,
        ]

        if not np.isnan(
            value
        ):

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

fig.colorbar(
    image,
    ax=ax,
)

fig.tight_layout()

subject_stage_plot_path = (
    results_dir
    / "stage34_subject_stage_f1.png"
)

plt.savefig(
    subject_stage_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. Save CSV outputs
# ============================================================

overall_class_path = (
    results_dir
    / "stage34_overall_per_class_metrics.csv"
)

overall_class_df.to_csv(
    overall_class_path,
    index=False,
)


subject_overall_path = (
    results_dir
    / "stage34_subject_overall_metrics.csv"
)

subject_overall_df.to_csv(
    subject_overall_path,
    index=False,
)


subject_stage_path = (
    results_dir
    / "stage34_subject_stage_metrics.csv"
)

subject_stage_df.to_csv(
    subject_stage_path,
    index=False,
)


subject_stage_pivot_path = (
    results_dir
    / "stage34_subject_stage_f1_matrix.csv"
)

subject_stage_f1_pivot.to_csv(
    subject_stage_pivot_path,
)


weakest_subject_stage_path = (
    results_dir
    / "stage34_weakest_stage_by_subject.csv"
)

weakest_subject_stage_df.to_csv(
    weakest_subject_stage_path,
    index=False,
)


confusion_matrix_path = (
    results_dir
    / "stage34_confusion_matrix.csv"
)

cm_df.to_csv(
    confusion_matrix_path,
)


normalized_confusion_path = (
    results_dir
    / "stage34_confusion_matrix_normalized.csv"
)

cm_normalized_df.to_csv(
    normalized_confusion_path,
)


error_pairs_path = (
    results_dir
    / "stage34_error_pairs.csv"
)

error_pair_df.to_csv(
    error_pairs_path,
    index=False,
)


subject_confusions_path = (
    results_dir
    / "stage34_subject_confusion_pairs.csv"
)

subject_confusion_df.to_csv(
    subject_confusions_path,
    index=False,
)


support_path = (
    results_dir
    / "stage34_class_distribution.csv"
)

support_df.to_csv(
    support_path,
    index=False,
)


# ============================================================
# 22. Compact Stage 34 summary
# ============================================================

if len(
    error_pair_df
) > 0:

    top_error = (
        error_pair_df.iloc[
            0
        ]
    )

    top_error_true = str(
        top_error[
            "true_stage"
        ]
    )

    top_error_pred = str(
        top_error[
            "predicted_stage"
        ]
    )

    top_error_count = int(
        top_error[
            "error_count"
        ]
    )

    top_error_percent = float(
        top_error[
            "percent_of_all_errors"
        ]
    )

else:

    top_error_true = ""
    top_error_pred = ""
    top_error_count = 0
    top_error_percent = 0.0


summary_df = pd.DataFrame([
    {
        "total_epochs":
            len(df),

        "subject_count":
            len(subjects),

        "raw_macro_f1":
            raw_overall[
                "macro_f1"
            ],

        "smoothed_macro_f1":
            smoothed_overall[
                "macro_f1"
            ],

        "weakest_stage":
            weakest_stage,

        "weakest_stage_f1":
            weakest_stage_f1,

        "worst_subject":
            worst_subject,

        "worst_subject_macro_f1":
            worst_subject_f1,

        "top_confusion_true":
            top_error_true,

        "top_confusion_predicted":
            top_error_pred,

        "top_confusion_count":
            top_error_count,

        "top_confusion_percent":
            top_error_percent,
    }
])

summary_path = (
    results_dir
    / "stage34_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# 23. Final interpretation
# ============================================================

print("\n" + "=" * 82)
print("STAGE 34 SUMMARY")
print("=" * 82)

print(
    "\nOverall Smoothed Macro F1:",
    round(
        smoothed_overall[
            "macro_f1"
        ],
        4,
    )
)

print(
    "\nWeakest sleep stage:",
    weakest_stage
)

print(
    "Weakest stage F1:",
    round(
        weakest_stage_f1,
        4,
    )
)

print(
    "\nWorst subject:",
    worst_subject
)

print(
    "Worst subject Macro F1:",
    round(
        worst_subject_f1,
        4,
    )
)

if len(
    error_pair_df
) > 0:

    print(
        "\nMost common confusion:"
    )

    print(
        f"{top_error_true} -> "
        f"{top_error_pred}"
    )

    print(
        "Error count:",
        top_error_count
    )

    print(
        "Percent of all errors:",
        round(
            top_error_percent,
            2,
        ),
        "%"
    )


# ============================================================
# 24. Diagnostic guidance
# ============================================================

print("\nDiagnostic interpretation:")

if weakest_stage_f1 < 0.40:

    print(
        f"{weakest_stage} is a major performance bottleneck."
    )

    print(
        "Stage 35 should prioritize class-specific improvement "
        "rather than global model tuning."
    )

elif weakest_stage_f1 < 0.60:

    print(
        f"{weakest_stage} is the main weak class, "
        "but performance is not completely collapsed."
    )

    print(
        "Stage 35 should inspect class weighting, "
        "temporal context, and features for this stage."
    )

else:

    print(
        "No single class is catastrophically weak."
    )

    print(
        "Stage 35 should focus on subject variability "
        "and the dominant confusion pairs."
    )


# ============================================================
# 25. Output locations
# ============================================================

print("\nSaved outputs:")

output_paths = [
    overall_class_path,
    subject_overall_path,
    subject_stage_path,
    subject_stage_pivot_path,
    weakest_subject_stage_path,
    confusion_matrix_path,
    normalized_confusion_path,
    error_pairs_path,
    subject_confusions_path,
    support_path,
    summary_path,
    confusion_plot_path,
    class_f1_plot_path,
    subject_stage_plot_path,
]

for path in output_paths:

    print(path)


print("\n" + "=" * 82)
print("STAGE 34 COMPLETE")
print("=" * 82)
