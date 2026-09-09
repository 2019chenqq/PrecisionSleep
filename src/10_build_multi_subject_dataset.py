from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "sleep-edf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "multi_subject_features.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------
# 睡眠階段 mapping
# -------------------------

STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


# -------------------------
# EEG bands
# -------------------------

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}


def calculate_bandpower(signal, sfreq):

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

    # 避免除以 0
    if total_power <= 0:
        return None

    features = {}

    for band_name, (low, high) in BANDS.items():

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


def process_subject(subject_dir):

    recording_id = subject_dir.name

    # SC4001 → SC400
    # SC4002 → SC400
    subject_id = recording_id[:5]

    print("\n==============================")
    print("Processing:", subject_id)
    print("==============================")


    # 自動找 PSG
    psg_files = list(
        subject_dir.glob("*PSG.edf")
    )

    # 自動找 Hypnogram
    hyp_files = list(
        subject_dir.glob("*Hypnogram.edf")
    )


    if len(psg_files) != 1:
        print(
            f"Skipped {subject_id}: "
            f"found {len(psg_files)} PSG files"
        )
        return None

    if len(hyp_files) != 1:
        print(
            f"Skipped {subject_id}: "
            f"found {len(hyp_files)} Hypnogram files"
        )
        return None


    psg_path = psg_files[0]
    hyp_path = hyp_files[0]


    print("PSG:", psg_path.name)
    print("Hypnogram:", hyp_path.name)


    # -------------------------
    # 讀 PSG
    # -------------------------

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=True,
        verbose=False
    )


    if "EEG Fpz-Cz" not in raw.ch_names:

        print(
            f"Skipped {subject_id}: "
            "EEG Fpz-Cz not found"
        )

        return None


    eeg = raw.copy().pick(
        ["EEG Fpz-Cz"]
    )

    sfreq = eeg.info["sfreq"]


    # -------------------------
    # 讀 Hypnogram
    # -------------------------

    annotations = mne.read_annotations(
        hyp_path
    )


    # -------------------------
    # 建立 stage 查詢函式
    # -------------------------

    def get_stage_at_time(time_sec):

        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description
        ):

            if (
                onset
                <= time_sec
                < onset + duration
            ):

                return STAGE_MAP.get(
                    description,
                    None
                )

        return None


    # -------------------------
    # 切 30 秒 epochs
    # -------------------------

    epoch_length = 30

    recording_duration = (
        eeg.n_times / sfreq
    )

    rows = []


    for epoch_index, start_time in enumerate(
        np.arange(
            0,
            recording_duration - epoch_length,
            epoch_length
        )
    ):

        label = get_stage_at_time(
            start_time
        )

        if label is None:
            continue


        start_sample = int(
            start_time * sfreq
        )

        stop_sample = int(
            (start_time + epoch_length)
            * sfreq
        )


        data, _ = eeg[
            :,
            start_sample:stop_sample
        ]


        signal = data[0]


        features = calculate_bandpower(
            signal,
            sfreq
        )

        if features is None:
            continue


        row = {
    "subject": subject_id,
    "recording": recording_id,
    "epoch": epoch_index,
    "start_sec": start_time,
    "label": label,
    **features,
}

        rows.append(row)


    df = pd.DataFrame(rows)


    print("Epochs created:", len(df))

    if not df.empty:

        print(
            df["label"]
            .value_counts()
        )


    return df


# =========================
# MAIN
# =========================

all_subjects = []


subject_dirs = sorted(
    [
        p
        for p in DATA_DIR.iterdir()
        if p.is_dir()
    ]
)


print(
    "Subjects found:",
    [p.name for p in subject_dirs]
)


for subject_dir in subject_dirs:

    subject_df = process_subject(
        subject_dir
    )

    if (
        subject_df is not None
        and not subject_df.empty
    ):

        all_subjects.append(
            subject_df
        )


if not all_subjects:

    raise RuntimeError(
        "No valid subjects processed."
    )


combined_df = pd.concat(
    all_subjects,
    ignore_index=True
)


combined_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n==============================")
print("MULTI-SUBJECT DATASET CREATED")
print("==============================")


print("\nShape:")
print(combined_df.shape)


print("\nSubjects:")
print(
    combined_df["subject"]
    .value_counts()
)


print("\nStage counts:")
print(
    combined_df["label"]
    .value_counts()
)


print("\nSaved to:")
print(OUTPUT_PATH)