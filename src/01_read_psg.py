from pathlib import Path
import mne

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PSG 檔案位置
psg_path = PROJECT_ROOT / "data" / "sleep-edf" / "SC4001E0-PSG.edf"

print("Reading:")
print(psg_path)

# 讀取 EDF
raw = mne.io.read_raw_edf(
    psg_path,
    preload=False,
    verbose=False
)

print("\nPSG loaded successfully!")

print("\nChannels:")
for ch in raw.ch_names:
    print("-", ch)

print("\nSampling frequency:")
print(raw.info["sfreq"], "Hz")

print("\nRecording duration:")
duration_seconds = raw.n_times / raw.info["sfreq"]
print(duration_seconds, "seconds")
print(duration_seconds / 3600, "hours")