from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

psg_path = (
    PROJECT_ROOT
    / "data"
    / "sleep-edf"
    / "SC4001E0-PSG.edf"
)


raw = mne.io.read_raw_edf(
    psg_path,
    preload=True,
    verbose=False
)

eeg = raw.copy().pick(["EEG Fpz-Cz"])

sfreq = eeg.info["sfreq"]


segments = {
    "N1": 30630,
    "N2": 30750,
    "N3": 31140,
}


bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Sigma": (12, 16),
    "Beta": (16, 30),
}


for stage, start_time in segments.items():

    start_sample = int(start_time * sfreq)
    stop_sample = int((start_time + 30) * sfreq)

    data, _ = eeg[:, start_sample:stop_sample]

    signal = data[0]

    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=400
    )

    # 只計算 0.5–30 Hz
    total_mask = (
        (frequencies >= 0.5)
        & (frequencies <= 30)
    )

    total_power = np.trapezoid(
        psd[total_mask],
        frequencies[total_mask]
    )


    print(f"\n===== {stage} =====")

    for band_name, (low, high) in bands.items():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        band_power = np.trapezoid(
            psd[mask],
            frequencies[mask]
        )

        relative_power = (
            band_power / total_power * 100
        )

        print(
            f"{band_name:6s}: "
            f"{relative_power:6.2f}%"
        )