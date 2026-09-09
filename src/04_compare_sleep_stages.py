from pathlib import Path

import matplotlib.pyplot as plt
import mne


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
    "Wake": 3600,
    "N1": 30630,
    "N2": 30750,
    "N3": 31140,
}


for stage, start_time in segments.items():

    start_sample = int(start_time * sfreq)
    stop_sample = int((start_time + 30) * sfreq)

    data, times = eeg[:, start_sample:stop_sample]

    data_uv = data[0] * 1e6

    plt.figure(figsize=(12, 4))

    plt.plot(times, data_uv)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"{stage} - EEG Fpz-Cz - 30 seconds")

    plt.tight_layout()
    plt.show()