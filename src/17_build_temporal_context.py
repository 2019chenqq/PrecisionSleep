from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_enhanced_features_trimmed.csv"
)

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_temporal_context.csv"
)


df = pd.read_csv(input_path)


# ==========================================
# 1. 原始 enhanced features
# ==========================================

feature_columns = [
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta",

    "rms",
    "std",
    "peak_to_peak",
    "zero_crossing_rate",

    "spectral_entropy",
    "dominant_frequency",
    "sef90",

    "delta_theta_ratio",
    "alpha_theta_ratio",
    "sigma_theta_ratio",
]


# ==========================================
# 2. 每個 recording 分開建立 temporal context
# ==========================================

context_recordings = []


for recording_id, group in df.groupby("recording"):

    group = group.sort_values(
        "start_sec"
    ).reset_index(drop=True)


    print("\n==============================")
    print("Processing:", recording_id)
    print("==============================")


    # --------------------------------------
    # Current epoch
    # --------------------------------------

    for feature in feature_columns:

        group[
            f"current_{feature}"
        ] = group[feature]


    # --------------------------------------
    # Previous 1 epoch = 前 30 秒
    # --------------------------------------

    for feature in feature_columns:

        group[
            f"prev1_{feature}"
        ] = group[feature].shift(1)


    # --------------------------------------
    # Previous 2 epochs = 前 60 秒
    # --------------------------------------

    for feature in feature_columns:

        group[
            f"prev2_{feature}"
        ] = group[feature].shift(2)


    # 前兩筆沒有完整 90 秒 context
    before = len(group)

    group = group.dropna().copy()

    after = len(group)


    print(
        f"Epochs: {before} -> {after}"
    )


    context_recordings.append(
        group
    )


# ==========================================
# 3. 合併全部 recordings
# ==========================================

context_df = pd.concat(
    context_recordings,
    ignore_index=True
)


# ==========================================
# 4. 只保留需要的欄位
# ==========================================

metadata_columns = [
    "subject",
    "recording",
    "epoch",
    "start_sec",
    "label",
]


temporal_feature_columns = []


for prefix in [
    "prev2",
    "prev1",
    "current",
]:

    for feature in feature_columns:

        temporal_feature_columns.append(
            f"{prefix}_{feature}"
        )


final_columns = (
    metadata_columns
    + temporal_feature_columns
)


context_df = context_df[
    final_columns
]


# ==========================================
# 5. 儲存
# ==========================================

context_df.to_csv(
    output_path,
    index=False
)


print("\n==============================")
print("TEMPORAL CONTEXT DATASET CREATED")
print("==============================")


print("\nShape:")
print(
    context_df.shape
)


print("\nSubjects:")
print(
    context_df[
        "subject"
    ].value_counts()
)


print("\nRecordings:")
print(
    context_df[
        "recording"
    ].value_counts()
)


print("\nFeature count:")
print(
    len(
        temporal_feature_columns
    )
)


print("\nFirst 10 temporal features:")
print(
    temporal_feature_columns[:10]
)


print("\nSaved to:")
print(
    output_path
)