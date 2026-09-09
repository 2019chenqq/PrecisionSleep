from pathlib import Path

import matplotlib.pyplot as plt
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


for stage, start_time in segments.items():

    start_sample = int(start_time * sfreq)
    stop_sample = int((start_time + 30) * sfreq)

    data, _ = eeg[:, start_sample:stop_sample]

    signal = data[0]

    # Welch PSD
    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=400
    )

    # 只看睡眠 EEG 主要頻段：0.5–30 Hz
    mask = (frequencies >= 0.5) & (frequencies <= 30)

    plt.figure(figsize=(10, 4))

    plt.plot(
        frequencies[mask],
        psd[mask]
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"{stage} - EEG Power Spectrum")

    plt.tight_layout()
    plt.show()