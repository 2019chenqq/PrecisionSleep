from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

psg_path = (
    PROJECT_ROOT
    / "data"
    / "sleep-edf"
    / "SC4001E0-PSG.edf"
)

hypnogram_path = (
    PROJECT_ROOT
    / "data"
    / "sleep-edf"
    / "SC4001EC-Hypnogram.edf"
)

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "SC4001_epoch_features.csv"
)


# -------------------------
# 1. 讀取 PSG
# -------------------------

raw = mne.io.read_raw_edf(
    psg_path,
    preload=True,
    verbose=False
)

eeg = raw.copy().pick(["EEG Fpz-Cz"])

sfreq = eeg.info["sfreq"]


# -------------------------
# 2. 讀取 Hypnogram
# -------------------------

annotations = mne.read_annotations(
    hypnogram_path
)


# -------------------------
# 3. 睡眠階段轉換
# -------------------------

stage_map = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


# -------------------------
# 4. EEG 頻段
# -------------------------

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}


# -------------------------
# 5. 找某個時間點的 sleep stage
# -------------------------

def get_stage_at_time(time_sec):

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description
    ):

        if onset <= time_sec < onset + duration:

            return stage_map.get(
                description,
                None
            )

    return None


# -------------------------
# 6. 計算 band power
# -------------------------

def calculate_bandpower(signal):

    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=400
    )

    total_mask = (
        (frequencies >= 0.5)
        & (frequencies <= 30)
    )

    total_power = np.trapezoid(
        psd[total_mask],
        frequencies[total_mask]
    )

    features = {}

    for band_name, (low, high) in bands.items():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        band_power = np.trapezoid(
            psd[mask],
            frequencies[mask]
        )

        features[band_name] = (
            band_power / total_power
        )

    return features


# -------------------------
# 7. 整晚切 30 秒 epoch
# -------------------------

epoch_length = 30

recording_duration = (
    eeg.n_times / sfreq
)

rows = []

epoch_index = 0

for start_time in np.arange(
    0,
    recording_duration - epoch_length,
    epoch_length
):

    label = get_stage_at_time(
        start_time
    )

    # 跳過不需要的標註
    if label is None:
        continue

    start_sample = int(
        start_time * sfreq
    )

    stop_sample = int(
        (start_time + epoch_length) * sfreq
    )

    data, _ = eeg[
        :,
        start_sample:stop_sample
    ]

    signal = data[0]

    features = calculate_bandpower(
        signal
    )

    row = {
        "epoch": epoch_index,
        "start_sec": start_time,
        "label": label,
        **features,
    }

    rows.append(row)

    epoch_index += 1


# -------------------------
# 8. 存成 CSV
# -------------------------

df = pd.DataFrame(rows)

df.to_csv(
    output_path,
    index=False
)


print("\nDataset created successfully!")

print("\nShape:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10))

print("\nSleep stage counts:")
print(df["label"].value_counts())

print("\nSaved to:")
print(output_path)