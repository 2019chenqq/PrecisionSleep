from pathlib import Path
import mne


PROJECT_ROOT = Path(__file__).resolve().parent.parent

hypnogram_path = (
    PROJECT_ROOT
    / "data"
    / "sleep-edf"
    / "SC4001EC-Hypnogram.edf"
)


# 讀取人工睡眠分期標註
annotations = mne.read_annotations(hypnogram_path)


print("Hypnogram loaded successfully!\n")

print("Number of annotations:")
print(len(annotations))


print("\nFirst 20 annotations:")

for i in range(min(20, len(annotations))):
    print(
        f"{i:02d} | "
        f"onset={annotations.onset[i]:8.1f}s | "
        f"duration={annotations.duration[i]:6.1f}s | "
        f"{annotations.description[i]}"
    )

    target_time = 3600

print(f"\nSleep stage at {target_time} seconds:")

for onset, duration, description in zip(
    annotations.onset,
    annotations.duration,
    annotations.description
):
    if onset <= target_time < onset + duration:
        print(description)
        print(f"Annotation range: {onset:.1f}s - {onset + duration:.1f}s")
        break