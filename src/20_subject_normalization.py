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

# ============================================================
# Stage 20 - Subject / Recording Relative Normalization
# ============================================================

normalized_df = df.copy()

for feature in feature_columns:

    normalized_df[feature] = (
        normalized_df
        .groupby("recording")[feature]
        .transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    )

df = normalized_df

print("\nSubject-relative normalization completed.")

print(
    "NaN count after normalization:",
    df[feature_columns].isna().sum().sum()
)

# ==========================================
# 2. LOSO
# ==========================================

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