"""
PrecisionSleep
Demo Stage 3 — Competition Visualization

Purpose
-------
Keep the verified Stage 2 inference pipeline unchanged,
but improve the competition presentation.

Outputs
-------
1. demo_predictions.csv
2. demo_eeg_preview.png
3. demo_reference_hypnogram.png
4. demo_prediction_hypnogram.png
5. demo_competition_view.png

Pipeline
--------
Raw Sleep-EDF EEG
    ↓
15 EEG features / 30-sec epoch
    ↓
Stage 15 analysis-window trimming
    ↓
90-sec temporal context
(prev2 + prev1 + current)
    ↓
45 canonical features
    ↓
Stage 43 global normalization
    ↓
Random Forest (150 trees)
    ↓
N1-preserving Viterbi
    ↓
Wake / N1 / N2 / N3 / REM
"""

from pathlib import Path
import json
import time

import joblib
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch


# ============================================================
# 0. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEMO_RECORDING = "SC4001"

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "sleep-edf"
    / DEMO_RECORDING
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage43_canonical"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "demo"
    / "demo_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIR
    / "stage43_canonical_random_forest.joblib"
)

MANIFEST_PATH = (
    MODEL_DIR
    / "stage43_canonical_manifest.json"
)

FEATURE_SCHEMA_PATH = (
    MODEL_DIR
    / "stage43_feature_schema.csv"
)

NORMALIZATION_PATH = (
    MODEL_DIR
    / "stage43_normalization_statistics.csv"
)

START_PROB_PATH = (
    MODEL_DIR
    / "stage43_start_probabilities.csv"
)

TRANSITION_PATH = (
    MODEL_DIR
    / "stage43_transition_matrix.csv"
)


# ============================================================
# 1. Canonical constants
# ============================================================

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


BASE_FEATURES = [
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta",

    "rms",
    "std",
    "peak_to_peak",
    "zero_crossing_rate",

    "spectral_entropy",
    "dominant_frequency",
    "sef90",

    "delta_theta_ratio",
    "alpha_theta_ratio",
    "sigma_theta_ratio",
]


EPOCH_LENGTH_SEC = 30

# Stage 15:
# 30 minutes of Wake retained before and after sleep.
TRIM_MARGIN_EPOCHS = 60


# ============================================================
# 2. Console helpers
# ============================================================

def print_header():

    print("\n" + "=" * 76)
    print(" PrecisionSleep")
    print(" COMPETITION DEMO — REAL SLEEP-STAGING PIPELINE")
    print("=" * 76)

    print(
        "\nRaw EEG → 15 EEG Features → 90-sec Context "
        "→ 45 Features → RF → Viterbi"
    )


def print_step(
    number,
    total,
    title,
):

    print("\n" + "=" * 76)

    print(
        f"[STEP {number}/{total}] "
        f"{title}"
    )

    print("=" * 76)


# ============================================================
# 3. Locate recording files
# ============================================================

def find_recording_files():

    psg_files = list(
        DATA_DIR.glob("*PSG.edf")
    )

    hyp_files = list(
        DATA_DIR.glob("*Hypnogram.edf")
    )

    if len(psg_files) != 1:

        raise RuntimeError(
            f"Expected exactly one PSG file in {DATA_DIR}, "
            f"found {len(psg_files)}."
        )

    if len(hyp_files) != 1:

        raise RuntimeError(
            f"Expected exactly one Hypnogram file in {DATA_DIR}, "
            f"found {len(hyp_files)}."
        )

    return (
        psg_files[0],
        hyp_files[0],
    )


# ============================================================
# 4. Load Stage 43 canonical package
# ============================================================

def load_canonical_package():

    print_step(
        1,
        7,
        "Load Stage 43 Canonical Package",
    )

    required_files = [
        MODEL_PATH,
        MANIFEST_PATH,
        FEATURE_SCHEMA_PATH,
        NORMALIZATION_PATH,
        START_PROB_PATH,
        TRANSITION_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing canonical artifact:\n{path}"
            )

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    feature_schema = pd.read_csv(
        FEATURE_SCHEMA_PATH
    )

    normalization = pd.read_csv(
        NORMALIZATION_PATH
    )

    start_df = pd.read_csv(
        START_PROB_PATH
    )

    transition_df = pd.read_csv(
        TRANSITION_PATH,
        index_col=0,
    )

    canonical_features = (
        feature_schema
        .sort_values("feature_index")["feature"]
        .tolist()
    )

    stage_order = manifest[
        "stage_order"
    ]

    if len(canonical_features) != 45:

        raise RuntimeError(
            "Expected 45 canonical features."
        )

    if hasattr(
        model,
        "feature_names_in_",
    ):

        if (
            list(model.feature_names_in_)
            != canonical_features
        ):

            raise RuntimeError(
                "Model feature order does not match "
                "Stage 43 feature schema."
            )

    if (
        normalization["feature"].tolist()
        != canonical_features
    ):

        raise RuntimeError(
            "Normalization feature order does not match "
            "Stage 43 feature schema."
        )

    print(
        "\nModel:",
        manifest["model_family"]
    )

    print(
        "Trees:",
        manifest["n_estimators"]
    )

    print(
        "Features:",
        len(canonical_features)
    )

    print(
        "Decoder:",
        manifest["decoder"]
    )

    print(
        "Transition weight:",
        manifest[
            "canonical_transition_weight"
        ]
    )

    print(
        "N1 emission boost:",
        manifest[
            "canonical_n1_emission_boost"
        ]
    )

    return {
        "model":
            model,

        "manifest":
            manifest,

        "canonical_features":
            canonical_features,

        "normalization":
            normalization,

        "start_df":
            start_df,

        "transition_df":
            transition_df,

        "stage_order":
            stage_order,
    }


# ============================================================
# 5. Exact Stage 14 feature extraction
# ============================================================

def calculate_features(
    signal,
    sfreq,
):

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    std = np.std(
        signal
    )

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


    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=400,
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
        freq_valid,
    )

    if (
        not np.isfinite(total_power)
        or total_power <= 0
    ):

        return None


    features = {}
    band_values = {}

    for band_name, (
        low,
        high,
    ) in BANDS.items():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        band_power = np.trapezoid(
            psd[mask],
            frequencies[mask],
        )

        relative_power = (
            band_power
            / total_power
        )

        features[
            band_name
        ] = relative_power

        band_values[
            band_name
        ] = relative_power


    psd_sum = np.sum(
        psd_valid
    )

    if psd_sum <= 0:

        return None

    psd_normalized = (
        psd_valid
        / psd_sum
    )

    psd_normalized = (
        psd_normalized[
            psd_normalized > 0
        ]
    )

    spectral_entropy = -np.sum(
        psd_normalized
        * np.log2(
            psd_normalized
        )
    )


    dominant_frequency = (
        freq_valid[
            np.argmax(
                psd_valid
            )
        ]
    )


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


    features.update({
        "rms":
            rms,

        "std":
            std,

        "peak_to_peak":
            peak_to_peak,

        "zero_crossing_rate":
            zero_crossing_rate,

        "spectral_entropy":
            spectral_entropy,

        "dominant_frequency":
            dominant_frequency,

        "sef90":
            sef90,

        "delta_theta_ratio":
            delta_theta_ratio,

        "alpha_theta_ratio":
            alpha_theta_ratio,

        "sigma_theta_ratio":
            sigma_theta_ratio,
    })

    return features


# ============================================================
# 6. Raw EEG → 15 enhanced features
# ============================================================

def build_enhanced_features():

    print_step(
        2,
        7,
        "Raw EEG → 30-second Feature Extraction",
    )

    (
        psg_path,
        hypnogram_path,
    ) = find_recording_files()

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=True,
        verbose=False,
    )

    if (
        "EEG Fpz-Cz"
        not in raw.ch_names
    ):

        raise RuntimeError(
            "EEG Fpz-Cz channel not found."
        )

    eeg = raw.copy().pick(
        ["EEG Fpz-Cz"]
    )

    sfreq = float(
        eeg.info["sfreq"]
    )

    annotations = (
        mne.read_annotations(
            hypnogram_path
        )
    )


    def get_stage_at_time(
        time_sec,
    ):

        for (
            onset,
            duration,
            description,
        ) in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        ):

            if (
                onset
                <= time_sec
                < onset + duration
            ):

                return STAGE_MAP.get(
                    description,
                    None,
                )

        return None


    recording_duration = (
        eeg.n_times
        / sfreq
    )

    rows = []

    start_times = np.arange(
        0,
        recording_duration
        - EPOCH_LENGTH_SEC,
        EPOCH_LENGTH_SEC,
    )

    extraction_start = (
        time.perf_counter()
    )


    for epoch_index, start_time in enumerate(
        start_times
    ):

        label = get_stage_at_time(
            start_time
        )

        if label is None:

            continue

        start_sample = int(
            start_time
            * sfreq
        )

        stop_sample = int(
            (
                start_time
                + EPOCH_LENGTH_SEC
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
            sfreq,
        )

        if features is None:

            continue

        rows.append({
            "subject":
                DEMO_RECORDING[:5],

            "recording":
                DEMO_RECORDING,

            "epoch":
                epoch_index,

            "start_sec":
                float(start_time),

            "label":
                label,

            **features,
        })


    enhanced_df = pd.DataFrame(
        rows
    )

    if enhanced_df.empty:

        raise RuntimeError(
            "No valid epochs generated."
        )


    elapsed = (
        time.perf_counter()
        - extraction_start
    )


    print(
        "\nRecording:",
        DEMO_RECORDING
    )

    print(
        "EEG channel:",
        "EEG Fpz-Cz"
    )

    print(
        "Sampling rate:",
        f"{sfreq:g} Hz"
    )

    print(
        "Epoch length:",
        "30 sec"
    )

    print(
        "Base features:",
        len(BASE_FEATURES)
    )

    print(
        "Epochs extracted:",
        len(enhanced_df)
    )

    print(
        "Feature extraction time:",
        f"{elapsed:.2f} sec"
    )

    return (
        enhanced_df,
        eeg,
        sfreq,
    )


# ============================================================
# 7. Stage 15 trimming
# ============================================================

def trim_reference_window(
    df,
):

    print_step(
        3,
        7,
        "Build Offline Analysis Window",
    )

    working = (
        df
        .sort_values("start_sec")
        .reset_index(drop=True)
    )

    sleep_rows = working[
        working["label"] != "Wake"
    ]

    if sleep_rows.empty:

        raise RuntimeError(
            "No sleep epochs found."
        )

    first_sleep = int(
        sleep_rows.index.min()
    )

    last_sleep = int(
        sleep_rows.index.max()
    )

    start_index = max(
        0,
        first_sleep
        - TRIM_MARGIN_EPOCHS,
    )

    end_index = min(
        len(working) - 1,
        last_sleep
        + TRIM_MARGIN_EPOCHS,
    )

    trimmed = (
        working
        .loc[
            start_index:end_index
        ]
        .copy()
        .reset_index(drop=True)
    )


    print(
        "\nOriginal epochs:",
        len(working)
    )

    print(
        "Analysis-window epochs:",
        len(trimmed)
    )

    print(
        "Rule:",
        "30 min before first sleep → "
        "30 min after last sleep"
    )

    return trimmed


# ============================================================
# 8. Stage 17 temporal context
# ============================================================

def build_temporal_context(
    df,
):

    print_step(
        4,
        7,
        "Build 90-second Temporal Context",
    )

    group = (
        df
        .sort_values("start_sec")
        .reset_index(drop=True)
        .copy()
    )


    for feature in BASE_FEATURES:

        group[
            f"current_{feature}"
        ] = group[
            feature
        ]


    for feature in BASE_FEATURES:

        group[
            f"prev1_{feature}"
        ] = (
            group[
                feature
            ]
            .shift(1)
        )


    for feature in BASE_FEATURES:

        group[
            f"prev2_{feature}"
        ] = (
            group[
                feature
            ]
            .shift(2)
        )


    before = len(group)

    group = (
        group
        .dropna()
        .copy()
        .reset_index(drop=True)
    )

    after = len(group)


    print(
        "\nEpochs:",
        f"{before} → {after}"
    )

    print(
        "Temporal context:",
        "Previous 60 sec + Current 30 sec"
    )

    print(
        "Feature dimensions:",
        "15 × 3 = 45"
    )

    return group


# ============================================================
# 9. Canonical normalization
# ============================================================

def normalize_features(
    context_df,
    package,
):

    print_step(
        5,
        7,
        "Apply Stage 43 Canonical Normalization",
    )

    features = package[
        "canonical_features"
    ]

    normalization = (
        package[
            "normalization"
        ]
        .set_index("feature")
    )

    missing = [
        feature
        for feature in features
        if feature not in context_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing canonical features:\n"
            + "\n".join(missing)
        )


    normalized = (
        context_df.copy()
    )


    for feature in features:

        mean_value = float(
            normalization.loc[
                feature,
                "mean",
            ]
        )

        std_value = float(
            normalization.loc[
                feature,
                "std",
            ]
        )

        normalized[
            feature
        ] = (
            normalized[
                feature
            ].astype(float)
            - mean_value
        ) / (
            std_value
            + 1e-8
        )


    normalized[
        features
    ] = (
        normalized[
            features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )


    print(
        "\nNormalization:",
        "training-global mean/std"
    )

    print(
        "Model input:",
        normalized[
            features
        ].shape
    )

    return normalized


# ============================================================
# 10. Probability alignment
# ============================================================

def align_model_probabilities(
    model,
    probabilities,
    stage_order,
):

    aligned = np.full(
        (
            len(probabilities),
            len(stage_order),
        ),
        1e-12,
        dtype=float,
    )

    class_to_idx = {
        label: idx
        for idx, label
        in enumerate(model.classes_)
    }


    for stage_idx, stage in enumerate(
        stage_order
    ):

        if stage in class_to_idx:

            aligned[
                :,
                stage_idx
            ] = probabilities[
                :,
                class_to_idx[
                    stage
                ],
            ]


    aligned = np.clip(
        aligned,
        1e-12,
        1.0,
    )

    aligned = (
        aligned
        / aligned.sum(
            axis=1,
            keepdims=True,
        )
    )

    return aligned


# ============================================================
# 11. N1-preserving Viterbi
# ============================================================

def viterbi_decode_sequence(
    emission_probabilities,
    start_probabilities,
    transition_probabilities,
    transition_weight,
    n1_emission_boost,
    stage_order,
):

    n_epochs = (
        emission_probabilities.shape[0]
    )

    n_states = (
        emission_probabilities.shape[1]
    )

    n1_index = (
        stage_order.index("N1")
    )


    log_emission = np.log(
        np.clip(
            emission_probabilities,
            1e-12,
            1.0,
        )
    )

    log_emission[
        :,
        n1_index
    ] += np.log(
        n1_emission_boost
    )


    log_start = np.log(
        np.clip(
            start_probabilities,
            1e-12,
            1.0,
        )
    )

    log_transition = np.log(
        np.clip(
            transition_probabilities,
            1e-12,
            1.0,
        )
    )


    dp = np.full(
        (
            n_epochs,
            n_states,
        ),
        -np.inf,
        dtype=float,
    )

    backpointer = np.zeros(
        (
            n_epochs,
            n_states,
        ),
        dtype=int,
    )


    dp[
        0,
        :
    ] = (
        log_start
        + log_emission[
            0,
            :
        ]
    )


    for t in range(
        1,
        n_epochs,
    ):

        for current_state in range(
            n_states
        ):

            transition_scores = (
                dp[
                    t - 1,
                    :
                ]
                + transition_weight
                * log_transition[
                    :,
                    current_state
                ]
            )

            best_previous_state = int(
                np.argmax(
                    transition_scores
                )
            )

            dp[
                t,
                current_state
            ] = (
                transition_scores[
                    best_previous_state
                ]
                + log_emission[
                    t,
                    current_state
                ]
            )

            backpointer[
                t,
                current_state
            ] = (
                best_previous_state
            )


    best_path = np.zeros(
        n_epochs,
        dtype=int,
    )

    best_path[-1] = int(
        np.argmax(
            dp[-1, :]
        )
    )


    for t in range(
        n_epochs - 2,
        -1,
        -1,
    ):

        best_path[t] = (
            backpointer[
                t + 1,
                best_path[t + 1],
            ]
        )


    return best_path


# ============================================================
# 12. Stage 43 inference
# ============================================================

def run_canonical_inference(
    normalized_df,
    package,
):

    print_step(
        6,
        7,
        "Run Stage 43 Canonical Inference",
    )

    model = package[
        "model"
    ]

    manifest = package[
        "manifest"
    ]

    features = package[
        "canonical_features"
    ]

    stage_order = package[
        "stage_order"
    ]


    X = normalized_df[
        features
    ]


    inference_start = (
        time.perf_counter()
    )


    raw_probabilities = (
        model.predict_proba(
            X
        )
    )


    aligned_probabilities = (
        align_model_probabilities(
            model,
            raw_probabilities,
            stage_order,
        )
    )


    start_lookup = (
        package[
            "start_df"
        ]
        .set_index("stage")[
            "start_probability"
        ]
    )


    start_probabilities = np.array(
        [
            float(
                start_lookup.loc[
                    stage
                ]
            )
            for stage in stage_order
        ],
        dtype=float,
    )


    transition_probabilities = (
        package[
            "transition_df"
        ]
        .loc[
            stage_order,
            stage_order,
        ]
        .to_numpy(
            dtype=float
        )
    )


    transition_weight = float(
        manifest[
            "canonical_transition_weight"
        ]
    )

    n1_boost = float(
        manifest[
            "canonical_n1_emission_boost"
        ]
    )


    path = viterbi_decode_sequence(
        aligned_probabilities,
        start_probabilities,
        transition_probabilities,
        transition_weight,
        n1_boost,
        stage_order,
    )


    predictions = np.array(
        [
            stage_order[index]
            for index in path
        ],
        dtype=object,
    )


    inference_elapsed = (
        time.perf_counter()
        - inference_start
    )


    results = normalized_df[
        [
            "subject",
            "recording",
            "epoch",
            "start_sec",
            "label",
        ]
    ].copy()


    results[
        "prediction"
    ] = predictions


    for stage_idx, stage in enumerate(
        stage_order
    ):

        results[
            f"prob_{stage}"
        ] = aligned_probabilities[
            :,
            stage_idx
        ]


    # Competition visualization:
    # time begins at 0 within the analysis window.
    analysis_start_sec = float(
        results[
            "start_sec"
        ].iloc[0]
    )

    results[
        "relative_sec"
    ] = (
        results[
            "start_sec"
        ]
        - analysis_start_sec
    )

    results[
        "relative_hour"
    ] = (
        results[
            "relative_sec"
        ]
        / 3600
    )


    print(
        "\nPredicted epochs:",
        len(results)
    )

    print(
        "Inference time:",
        f"{inference_elapsed:.4f} sec"
    )

    print(
        "Decoder:",
        "N1-preserving Viterbi"
    )

    print(
        "Analysis timeline:",
        "0 h → "
        f"{results['relative_hour'].max():.2f} h"
    )

    return results


# ============================================================
# 13. Competition visualization
# ============================================================

STAGE_TO_Y = {
    "Wake": 4,
    "REM": 3,
    "N1": 2,
    "N2": 1,
    "N3": 0,
}


def save_eeg_preview(
    eeg,
    sfreq,
    results,
):

    analysis_start_sec = float(
        results[
            "start_sec"
        ].iloc[0]
    )

    preview_seconds = 60

    start_sample = int(
        analysis_start_sec
        * sfreq
    )

    stop_sample = int(
        (
            analysis_start_sec
            + preview_seconds
        )
        * sfreq
    )


    signal = eeg.get_data(
        start=start_sample,
        stop=stop_sample,
    )[0]


    time_axis = (
        np.arange(
            len(signal)
        )
        / sfreq
    )


    fig = plt.figure(
        figsize=(12, 3.6)
    )

    ax = fig.add_axes(
        [0.08, 0.20, 0.89, 0.68]
    )


    ax.plot(
        time_axis,
        signal,
        linewidth=0.8,
    )

    ax.set_title(
        "Raw EEG Input — EEG Fpz-Cz",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time from preview start (seconds)"
    )

    ax.set_ylabel(
        "Amplitude (V)"
    )

    ax.grid(
        alpha=0.15
    )


    output_path = (
        OUTPUT_DIR
        / "demo_eeg_preview.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def save_single_hypnogram(
    results,
    column,
    title,
    filename,
):

    time_hours = (
        results[
            "relative_hour"
        ].to_numpy()
    )

    y = np.array(
        [
            STAGE_TO_Y[
                stage
            ]
            for stage in results[
                column
            ]
        ]
    )


    fig = plt.figure(
        figsize=(13, 4)
    )

    ax = fig.add_axes(
        [0.07, 0.19, 0.91, 0.70]
    )


    ax.step(
        time_hours,
        y,
        where="post",
        linewidth=1.35,
    )


    ax.set_yticks(
        [4, 3, 2, 1, 0]
    )

    ax.set_yticklabels(
        [
            "Wake",
            "REM",
            "N1",
            "N2",
            "N3",
        ]
    )


    ax.set_xlim(
        0,
        max(
            time_hours.max(),
            0.1,
        )
    )


    ax.set_xlabel(
        "Time from analysis-window start (hours)"
    )

    ax.set_ylabel(
        "Sleep stage"
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        alpha=0.15,
    )


    output_path = (
        OUTPUT_DIR
        / filename
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def save_competition_view(
    results,
    package,
):

    manifest = package["manifest"]

    time_hours = results["relative_hour"].to_numpy()

    reference_y = np.array(
        [STAGE_TO_Y[stage] for stage in results["label"]]
    )

    prediction_y = np.array(
        [STAGE_TO_Y[stage] for stage in results["prediction"]]
    )

    max_hour = max(time_hours.max(), 0.1)

    fig = plt.figure(figsize=(16, 9))


    # ========================================================
    # Header
    # ========================================================
    fig.text(
        0.04,
        0.95,
        "PrecisionSleep",
        fontsize=28,
        fontweight="bold",
    )

    fig.text(
        0.04,
        0.91,
        "Lightweight EEG Sleep Staging for Wearable Deployment",
        fontsize=15,
    )


        # ========================================================
    # Left Information Panel
    # ========================================================
    ax_left = fig.add_axes([0.035, 0.11, 0.255, 0.72])
    ax_left.axis("off")

    info_text = (
        "PIPELINE\n\n"

        "RAW EEG\n"
        "EEG Fpz-Cz\n"
        "30-sec epochs\n\n"
        "↓\n\n"

        "15 EEG FEATURES\n"
        "Band power + time + spectral features\n\n"
        "↓\n\n"

        "90-sec CONTEXT\n"
        "Previous 60 s + Current 30 s\n\n"
        "↓\n\n"

        "45 FEATURES\n\n"
        "↓\n\n"

        "RANDOM FOREST\n"
        "150 trees\n\n"
        "↓\n\n"

        "N1-PRESERVING\n"
        "VITERBI DECODER\n\n"
        "↓\n\n"

        "5 SLEEP STAGES\n\n"

        "------------------------------\n\n"

        "CANONICAL MODEL\n\n"
        f"Features: 45\n"
        f"Trees: {manifest['n_estimators']}\n"
        f"Transition weight: {manifest['canonical_transition_weight']}\n"
        f"N1 boost: {manifest['canonical_n1_emission_boost']}\n\n"

        "VALIDATION\n\n"
        "Stage 40 Nested LOSO\n"
        f"Mean Macro F1: {manifest['stage40_nested_loso_mean_macro_f1']:.4f}\n"
        f"Pooled Macro F1: {manifest['stage40_nested_loso_pooled_macro_f1']:.4f}"
    )

    ax_left.text(
        0.03,
        0.98,
        info_text,
        va="top",
        fontsize=10.6,
        linespacing=1.12,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="lightgray",
        ),
    )


    # ========================================================
    # Right Top — Reference
    # ========================================================
    ax_ref = fig.add_axes([0.34, 0.56, 0.62, 0.28])

    ax_ref.step(
        time_hours,
        reference_y,
        where="post",
        linewidth=1.25,
    )

    ax_ref.set_title(
        "Reference Sleep Stages",
        fontsize=15,
        fontweight="bold",
    )

    ax_ref.set_yticks([4, 3, 2, 1, 0])
    ax_ref.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])

    ax_ref.set_xlim(0, max_hour)
    ax_ref.set_ylabel("Stage")
    ax_ref.grid(axis="x", alpha=0.15)
    ax_ref.tick_params(labelbottom=False)


    # ========================================================
    # Right Bottom — Prediction
    # ========================================================
    ax_pred = fig.add_axes([0.34, 0.20, 0.62, 0.28])

    ax_pred.step(
        time_hours,
        prediction_y,
        where="post",
        linewidth=1.25,
    )

    ax_pred.set_title(
        "PrecisionSleep Prediction",
        fontsize=15,
        fontweight="bold",
    )

    ax_pred.set_yticks([4, 3, 2, 1, 0])
    ax_pred.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])

    ax_pred.set_xlim(0, max_hour)
    ax_pred.set_xlabel("Time from analysis-window start (hours)")
    ax_pred.set_ylabel("Stage")
    ax_pred.grid(axis="x", alpha=0.15)


    # ========================================================
    # Footer
    # ========================================================
    fig.text(
        0.34,
        0.07,
        (
            "Pipeline demonstration using Sleep-EDF. "
            "Generalization performance is reported from "
            "Stage 40 nested leave-one-subject-out validation."
        ),
        fontsize=10,
    )

    output_path = (
        OUTPUT_DIR
        / "demo_competition_view.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# 14. Save all competition outputs
# ============================================================

def save_outputs(
    results,
    eeg,
    sfreq,
    package,
):

    print_step(
        7,
        7,
        "Generate Competition Visualization",
    )


    predictions_path = (
        OUTPUT_DIR
        / "demo_predictions.csv"
    )

    results.to_csv(
        predictions_path,
        index=False,
    )


    eeg_path = save_eeg_preview(
        eeg,
        sfreq,
        results,
    )


    reference_path = (
        save_single_hypnogram(
            results,
            "label",
            "Reference Sleep Stages",
            "demo_reference_hypnogram.png",
        )
    )


    prediction_path = (
        save_single_hypnogram(
            results,
            "prediction",
            "PrecisionSleep Prediction",
            "demo_prediction_hypnogram.png",
        )
    )


    competition_path = (
        save_competition_view(
            results,
            package,
        )
    )


    print(
        "\nPrediction distribution:"
    )

    print(
        results[
            "prediction"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nCompetition outputs created:"
    )

    for path in [
        predictions_path,
        eeg_path,
        reference_path,
        prediction_path,
        competition_path,
    ]:

        print(
            path
        )


# ============================================================
# 15. Main
# ============================================================

def main():

    total_start = (
        time.perf_counter()
    )

    print_header()


    package = (
        load_canonical_package()
    )


    (
        enhanced_df,
        eeg,
        sfreq,
    ) = build_enhanced_features()


    trimmed_df = (
        trim_reference_window(
            enhanced_df
        )
    )


    context_df = (
        build_temporal_context(
            trimmed_df
        )
    )


    normalized_df = (
        normalize_features(
            context_df,
            package,
        )
    )


    results = (
        run_canonical_inference(
            normalized_df,
            package,
        )
    )


    save_outputs(
        results,
        eeg,
        sfreq,
        package,
    )


    total_elapsed = (
        time.perf_counter()
        - total_start
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        " DEMO STAGE 3 COMPLETE"
    )

    print(
        "=" * 76
    )


    print(
        "\nRecording:",
        DEMO_RECORDING
    )

    print(
        "Real EEG input:",
        "YES"
    )

    print(
        "Real 15-feature extraction:",
        "YES"
    )

    print(
        "Real 90-sec temporal context:",
        "YES"
    )

    print(
        "Real 45-feature Stage 43 model:",
        "YES"
    )

    print(
        "N1-preserving Viterbi:",
        "YES"
    )

    print(
        "Competition visualization:",
        "YES"
    )

    print(
        "Total runtime:",
        f"{total_elapsed:.2f} sec"
    )


    print(
        "\nPrimary validation:"
    )

    print(
        "Nested LOSO Mean Macro F1 = "
        f"{package['manifest']['stage40_nested_loso_mean_macro_f1']:.4f}"
    )


    print(
        "\nMain competition image:"
    )

    print(
        OUTPUT_DIR
        / "demo_competition_view.png"
    )


if __name__ == "__main__":

    main()