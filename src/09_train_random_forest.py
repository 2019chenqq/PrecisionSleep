from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_path = (
    PROJECT_ROOT
    / "outputs"
    / "SC4001_epoch_features_trimmed.csv"
)


# -------------------------
# 1. 讀取資料
# -------------------------

df = pd.read_csv(data_path)


feature_columns = [
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta",
]

X = df[feature_columns]
y = df["label"]


print("Dataset shape:")
print(df.shape)

print("\nClass counts:")
print(y.value_counts())


# -------------------------
# 2. 切 train / test
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


print("\nTrain size:")
print(len(X_train))

print("\nTest size:")
print(len(X_test))


# -------------------------
# 3. 建立 Random Forest
# -------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)


# -------------------------
# 4. 訓練
# -------------------------

model.fit(
    X_train,
    y_train
)


# -------------------------
# 5. 預測
# -------------------------

y_pred = model.predict(X_test)


# -------------------------
# 6. 評估
# -------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


print("\n====================")
print("MODEL RESULTS")
print("====================")

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nBalanced Accuracy:")
print(round(balanced_accuracy, 4))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

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

print(cm_df)


# -------------------------
# 7. Feature importance
# -------------------------

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