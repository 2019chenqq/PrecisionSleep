from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_features_trimmed.csv"
)


# ==========================================
# 1. 讀取資料
# ==========================================

df = pd.read_csv(data_path)


feature_columns = [
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta",
]


# ==========================================
# 2. Subject-wise split
# ==========================================

train_subjects = [
    "SC400",
    "SC401",
]

test_subject = "SC402"


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


print("==============================")
print("SUBJECT-WISE VALIDATION")
print("==============================")

print("\nTrain subjects:")
print(train_subjects)

print("\nTest subject:")
print(test_subject)

print("\nTrain epochs:")
print(len(train_df))

print("\nTest epochs:")
print(len(test_df))


print("\nTrain stage counts:")
print(y_train.value_counts())

print("\nTest stage counts:")
print(y_test.value_counts())


# ==========================================
# 3. Random Forest
# ==========================================

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


# ==========================================
# 4. 預測 unseen subject
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# 5. Metrics
# ==========================================

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


print("\n==============================")
print("UNSEEN SUBJECT RESULTS")
print("==============================")


print("\nAccuracy:")
print(round(accuracy, 4))

print("\nBalanced Accuracy:")
print(round(balanced_accuracy, 4))

print("\nMacro F1:")
print(round(macro_f1, 4))


# ==========================================
# 6. Classification Report
# ==========================================

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
# 7. Confusion Matrix
# ==========================================

labels = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
]


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)


print("\nConfusion Matrix:")
print(cm_df)


# ==========================================
# 8. Feature Importance
# ==========================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print("\nFeature Importance:")
print(importance_df)