from pathlib import Path

import matplotlib.pyplot as plt
import mne


PROJECT_ROOT = Path(__file__).resolve().parent.parent
psg_path = PROJECT_ROOT / "data" / "sleep-edf" / "SC4001E0-PSG.edf"


# 讀取 PSG
raw = mne.io.read_raw_edf(
    psg_path,
    preload=True,
    verbose=False
)


# 只抓 Fpz-Cz EEG
eeg = raw.copy().pick(["EEG Fpz-Cz"])


# 取 recording 第 1 小時開始後的 30 秒
start_time = 3600
duration = 30

start_sample = int(start_time * eeg.info["sfreq"])
stop_sample = int((start_time + duration) * eeg.info["sfreq"])


data, times = eeg[:, start_sample:stop_sample]


# MNE 的 EEG 單位是 Volt
# 轉成 microvolt (µV)，比較容易閱讀
data_uv = data[0] * 1e6


plt.figure(figsize=(12, 4))

plt.plot(times, data_uv)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.title("EEG Fpz-Cz - 30 second segment")

plt.tight_layout()
plt.show()