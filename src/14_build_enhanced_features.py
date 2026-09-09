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
    / "multi_subject_enhanced_features.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}


def calculate_features(signal, sfreq):

    # -------------------------
    # Time-domain features
    # -------------------------

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    std = np.std(signal)

    peak_to_peak = (
        np.max(signal)
        - np.min(signal)
    )

    zero_crossings = np.sum(
        np.diff(
            np.signbit(signal)
        )
    )

    zero_crossing_rate = (
        zero_crossings
        / len(signal)
    )


    # -------------------------
    # PSD
    # -------------------------

    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=400
    )

    valid_mask = (
        (frequencies >= 0.5)
        & (frequencies <= 30)
    )

    freq_valid = frequencies[
        valid_mask
    ]

    psd_valid = psd[
        valid_mask
    ]

    total_power = np.trapezoid(
        psd_valid,
        freq_valid
    )

    if total_power <= 0:
        return None


    # -------------------------
    # Relative band power
    # -------------------------

    features = {}

    band_values = {}

    for band_name, (low, high) in BANDS.items():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        band_power = np.trapezoid(
            psd[mask],
            frequencies[mask]
        )

        relative_power = (
            band_power / total_power
        )

        features[band_name] = (
            relative_power
        )

        band_values[
            band_name
        ] = relative_power


    # -------------------------
    # Spectral entropy
    # -------------------------

    psd_normalized = (
        psd_valid
        / np.sum(psd_valid)
    )

    psd_normalized = (
        psd_normalized[
            psd_normalized > 0
        ]
    )

    spectral_entropy = -np.sum(
        psd_normalized
        * np.log2(psd_normalized)
    )


    # -------------------------
    # Dominant frequency
    # -------------------------

    dominant_frequency = (
        freq_valid[
            np.argmax(psd_valid)
        ]
    )


    # -------------------------
    # Spectral edge frequency 90%
    # -------------------------

    cumulative_power = np.cumsum(
        psd_valid
    )

    cumulative_power = (
        cumulative_power
        / cumulative_power[-1]
    )

    sef90_index = np.where(
        cumulative_power >= 0.90
    )[0][0]

    sef90 = freq_valid[
        sef90_index
    ]


    # -------------------------
    # Band ratios
    # -------------------------

    eps = 1e-12

    delta_theta_ratio = (
        band_values["delta"]
        /
        (
            band_values["theta"]
            + eps
        )
    )

    alpha_theta_ratio = (
        band_values["alpha"]
        /
        (
            band_values["theta"]
            + eps
        )
    )

    sigma_theta_ratio = (
        band_values["sigma"]
        /
        (
            band_values["theta"]
            + eps
        )
    )


    # -------------------------
    # Add features
    # -------------------------

    features.update({
        "rms": rms,
        "std": std,
        "peak_to_peak": peak_to_peak,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_entropy": spectral_entropy,
        "dominant_frequency": dominant_frequency,
        "sef90": sef90,
        "delta_theta_ratio": delta_theta_ratio,
        "alpha_theta_ratio": alpha_theta_ratio,
        "sigma_theta_ratio": sigma_theta_ratio,
    })

    return features


def process_recording(subject_dir):

    recording_id = subject_dir.name
    subject_id = recording_id[:5]

    print("\n==============================")
    print("Processing:", recording_id)
    print("Subject:", subject_id)
    print("==============================")


    psg_files = list(
        subject_dir.glob("*PSG.edf")
    )

    hyp_files = list(
        subject_dir.glob(
            "*Hypnogram.edf"
        )
    )


    if len(psg_files) != 1:
        print(
            f"Skipped {recording_id}: "
            f"found {len(psg_files)} PSG files"
        )
        return None

    if len(hyp_files) != 1:
        print(
            f"Skipped {recording_id}: "
            f"found {len(hyp_files)} Hypnogram files"
        )
        return None


    raw = mne.io.read_raw_edf(
        psg_files[0],
        preload=True,
        verbose=False
    )


    if "EEG Fpz-Cz" not in raw.ch_names:

        print(
            f"Skipped {recording_id}: "
            "EEG Fpz-Cz not found"
        )

        return None


    eeg = raw.copy().pick(
        ["EEG Fpz-Cz"]
    )

    sfreq = eeg.info["sfreq"]

    annotations = mne.read_annotations(
        hyp_files[0]
    )


    def get_stage_at_time(
        time_sec
    ):

        for (
            onset,
            duration,
            description
        ) in zip(
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
            (
                start_time
                + epoch_length
            )
            * sfreq
        )


        data, _ = eeg[
            :,
            start_sample:stop_sample
        ]

        signal = data[0]


        features = calculate_features(
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

    print(
        "Epochs created:",
        len(df)
    )

    return df


all_recordings = []


subject_dirs = sorted(
    [
        p
        for p in DATA_DIR.iterdir()
        if p.is_dir()
    ]
)


print(
    "Recordings found:",
    [
        p.name
        for p in subject_dirs
    ]
)


for subject_dir in subject_dirs:

    recording_df = process_recording(
        subject_dir
    )

    if (
        recording_df is not None
        and not recording_df.empty
    ):

        all_recordings.append(
            recording_df
        )


if not all_recordings:

    raise RuntimeError(
        "No valid recordings processed."
    )


combined_df = pd.concat(
    all_recordings,
    ignore_index=True
)


combined_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n==============================")
print("ENHANCED DATASET CREATED")
print("==============================")


print("\nShape:")
print(combined_df.shape)


print("\nSubjects:")
print(
    combined_df[
        "subject"
    ].value_counts()
)


print("\nFeature columns:")

feature_columns = [
    col
    for col in combined_df.columns
    if col not in [
        "subject",
        "recording",
        "epoch",
        "start_sec",
        "label",
    ]
]

print(feature_columns)


print("\nSaved to:")
print(OUTPUT_PATH)