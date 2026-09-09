from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_temporal_context.csv"
)


df = pd.read_csv(data_path)


# ==========================================
# 1. 自動找出 45 個 temporal features
# ==========================================

metadata_columns = [
    "subject",
    "recording",
    "epoch",
    "start_sec",
    "label",
]


feature_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]


subjects = sorted(
    df["subject"].unique()
)


print("==============================")
print("TEMPORAL LOSO VALIDATION")
print("==============================")

print("\nSubjects:")
print(subjects)

print("\nFeature count:")
print(len(feature_columns))


results = []


# ==========================================
# 2. LOSO
# ==========================================

all_y_true = []
all_y_pred = []
subject_error_rows = []

for test_subject in subjects:

    train_subjects = [
        s
        for s in subjects
        if s != test_subject
    ]


    train_df = df[
        df["subject"].isin(
            train_subjects
        )
    ].copy()

    test_df = df[
        df["subject"]
        == test_subject
    ].copy()


    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "label"
    ]


    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


    model.fit(
        X_train,
        y_train
    )


    y_pred = model.predict(
        X_test
    )

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())

    subject_report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    for stage in ["N1", "N2", "N3", "REM", "Wake"]:

        if stage in subject_report:

            subject_error_rows.append({
                "subject": test_subject,
                "stage": stage,
                "precision": subject_report[stage]["precision"],
                "recall": subject_report[stage]["recall"],
                "f1": subject_report[stage]["f1-score"],
                "support": subject_report[stage]["support"]
            })

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )


        results.append({
            "test_subject": test_subject,
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced_accuracy,
            "macro_f1": macro_f1,
        })


    print("\n==============================")
    print(
        "TEST SUBJECT:",
        test_subject
    )
    print("==============================")


    print("\nAccuracy:")
    print(
        round(
            accuracy,
            4
        )
    )


    print("\nBalanced Accuracy:")
    print(
        round(
            balanced_accuracy,
            4
        )
    )


    print("\nMacro F1:")
    print(
        round(
            macro_f1,
            4
        )
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=3,
            zero_division=0
        )
    )

from sklearn.metrics import classification_report, confusion_matrix

print("\n")
print("=" * 60)
print("OVERALL LOSO ERROR ANALYSIS")
print("=" * 60)

print("\nClassification Report:")
print(
    classification_report(
        all_y_true,
        all_y_pred,
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(
    all_y_true,
    all_y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ============================================
# Stage 19
# ============================================

import pandas as pd

subject_stage_df = pd.DataFrame(subject_error_rows)

print("\n")
print("=" * 60)
print("SUBJECT × STAGE F1 ANALYSIS")
print("=" * 60)

f1_table = subject_stage_df.pivot(
    index="subject",
    columns="stage",
    values="f1"
)

print("\nF1 by subject and sleep stage:")
print(f1_table.round(3))

print("\nMean F1 by stage:")
print(
    subject_stage_df
    .groupby("stage")["f1"]
    .mean()
    .sort_values()
    .round(4)
)

print("\nWorst subject-stage combinations:")

worst_cases = (
    subject_stage_df
    .sort_values("f1")
    .head(15)
)

print(
    worst_cases[
        [
            "subject",
            "stage",
            "precision",
            "recall",
            "f1",
            "support"
        ]
    ].round(4)
)

subject_stage_df.to_csv(
    "results/stage19_subject_stage_metrics.csv",
    index=False
)

f1_table.to_csv(
    "results/stage19_subject_stage_f1_table.csv"
)

print("\nStage 19 results saved.")

# ==========================================
# 3. Summary
# ==========================================

results_df = pd.DataFrame(
    results
)


print("\n\n==============================")
print("TEMPORAL LOSO SUMMARY")
print("==============================")


print("\nPer-subject results:")
print(results_df)


print("\nMean Accuracy:")
print(
    round(
        results_df[
            "accuracy"
        ].mean(),
        4
    )
)


print("\nMean Balanced Accuracy:")
print(
    round(
        results_df[
            "balanced_accuracy"
        ].mean(),
        4
    )
)


print("\nMean Macro F1:")
print(
    round(
        results_df[
            "macro_f1"
        ].mean(),
        4
    )
)


print("\nStandard deviation:")

print(
    results_df[
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]
    ].std().round(4)
)