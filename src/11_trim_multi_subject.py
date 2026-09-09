from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_features.csv"
)

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_features_trimmed.csv"
)


df = pd.read_csv(input_path)

trimmed_recordings = []

# 睡前與睡後各保留 30 分鐘 Wake
margin_epochs = 60


for recording_id, group in df.groupby("recording"):

    group = group.sort_values(
        "start_sec"
    ).reset_index(drop=True)

    sleep_rows = group[
        group["label"] != "Wake"
    ]

    if sleep_rows.empty:
        print(
            f"Skipped {recording_id}: "
            "no sleep epochs"
        )
        continue

    first_sleep = sleep_rows.index.min()
    last_sleep = sleep_rows.index.max()

    start_index = max(
        0,
        first_sleep - margin_epochs
    )

    end_index = min(
        len(group) - 1,
        last_sleep + margin_epochs
    )

    trimmed = group.loc[
        start_index:end_index
    ].copy()

    trimmed_recordings.append(
        trimmed
    )

    print(
        f"{recording_id}: "
        f"{len(group)} → {len(trimmed)} epochs"
    )


trimmed_df = pd.concat(
    trimmed_recordings,
    ignore_index=True
)


trimmed_df.to_csv(
    output_path,
    index=False
)


print("\n============================")
print("TRIMMED DATASET CREATED")
print("============================")

print("\nShape:")
print(trimmed_df.shape)

print("\nSubjects:")
print(
    trimmed_df["subject"]
    .value_counts()
)

print("\nRecordings:")
print(
    trimmed_df["recording"]
    .value_counts()
)

print("\nStage counts:")
print(
    trimmed_df["label"]
    .value_counts()
)

print("\nStage percentages:")
print(
    (
        trimmed_df["label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nSaved to:")
print(output_path)