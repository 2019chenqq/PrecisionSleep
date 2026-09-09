from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_features_trimmed.csv"
)


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv(data_path)


feature_columns = [
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta",
]


subjects = sorted(
    df["subject"].unique()
)


print("==============================")
print("LEAVE-ONE-SUBJECT-OUT")
print("==============================")

print("\nSubjects:")
print(subjects)


results = []


# ==========================================
# 2. LOSO loop
# ==========================================

for test_subject in subjects:

    train_subjects = [
        subject
        for subject in subjects
        if subject != test_subject
    ]


    train_df = df[
        df["subject"].isin(train_subjects)
    ].copy()

    test_df = df[
        df["subject"] == test_subject
    ].copy()


    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    X_test = test_df[feature_columns]
    y_test = test_df["label"]


    # ======================================
    # Model
    # ======================================

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


    # ======================================
    # Metrics
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )


    results.append({
        "test_subject": test_subject,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
    })


    print("\n==============================")
    print(f"TEST SUBJECT: {test_subject}")
    print("==============================")

    print(
        "Train subjects:",
        train_subjects
    )

    print(
        "Train epochs:",
        len(train_df)
    )

    print(
        "Test epochs:",
        len(test_df)
    )


    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nBalanced Accuracy:")
    print(round(balanced_accuracy, 4))

    print("\nMacro F1:")
    print(round(macro_f1, 4))


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=3,
            zero_division=0
        )
    )


# ==========================================
# 3. Summary
# ==========================================

results_df = pd.DataFrame(
    results
)


print("\n\n==============================")
print("LOSO SUMMARY")
print("==============================")


print("\nPer-subject results:")
print(results_df)


print("\nMean Accuracy:")
print(
    round(
        results_df["accuracy"].mean(),
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