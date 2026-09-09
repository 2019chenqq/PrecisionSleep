from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = (
    PROJECT_ROOT
    / "outputs"
    / "SC4001_epoch_features.csv"
)

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "SC4001_epoch_features_trimmed.csv"
)


df = pd.read_csv(input_path)


# 找到第一個與最後一個非 Wake epoch
sleep_rows = df[df["label"] != "Wake"]

first_sleep_index = sleep_rows.index.min()
last_sleep_index = sleep_rows.index.max()


# 30 分鐘 = 60 個 30 秒 epochs
margin_epochs = 60


start_index = max(
    0,
    first_sleep_index - margin_epochs
)

end_index = min(
    len(df) - 1,
    last_sleep_index + margin_epochs
)


trimmed_df = df.loc[
    start_index:end_index
].copy()


trimmed_df.to_csv(
    output_path,
    index=False
)


print("Trimmed dataset created!")

print("\nOriginal shape:")
print(df.shape)

print("\nTrimmed shape:")
print(trimmed_df.shape)

print("\nStage counts:")
print(trimmed_df["label"].value_counts())

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