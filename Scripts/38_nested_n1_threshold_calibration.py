from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "outputs" / "multi_subject_temporal_context.csv"
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 92)
print("STAGE 38 - NESTED N1 PROBABILITY THRESHOLD CALIBRATION")
print("=" * 92)

df_raw = pd.read_csv(data_path)
metadata_columns = ["subject", "recording", "epoch", "start_sec", "label"]
feature_columns = [c for c in df_raw.columns if c not in metadata_columns]
subjects = sorted(df_raw["subject"].unique())
stage_order = [s for s in ["Wake", "N1", "N2", "N3", "REM"] if s in set(df_raw["label"].unique())]
threshold_candidates = np.round(np.arange(0.10, 0.51, 0.025), 3).tolist()

print("\nSubjects:", subjects)
print("Feature count:", len(feature_columns))
print("Threshold candidates:", threshold_candidates)

def calculate_training_statistics(train_df, features):
    means, stds = {}, {}
    for feature in features:
        means[feature] = float(train_df[feature].mean())
        s = float(train_df[feature].std())
        stds[feature] = 1.0 if np.isnan(s) or s < 1e-8 else s
    return means, stds

def normalize_with_fixed_statistics(input_df, features, means, stds):
    out = input_df.copy()
    for feature in features:
        out[feature] = (out[feature].astype(float) - means[feature]) / (stds[feature] + 1e-8)
    out[features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def smooth_predictions_by_recording(metadata_df, predictions):
    out = metadata_df.copy()
    out["raw_prediction"] = predictions
    out["smoothed_prediction"] = predictions.copy()

    for recording, group in out.groupby("recording", sort=False):
        idx = group.index.to_list()
        pred = group["raw_prediction"].to_numpy().copy()
        smoothed = pred.copy()

        for i in range(1, len(pred) - 1):
            if pred[i - 1] == pred[i + 1] and pred[i] != pred[i - 1]:
                smoothed[i] = pred[i - 1]

        out.loc[idx, "smoothed_prediction"] = smoothed

    return out["smoothed_prediction"].to_numpy()

def overall_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

def per_class_metrics(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        label: {
            "precision": float(pp),
            "recall": float(rr),
            "f1": float(ff),
            "support": int(ss),
        }
        for label, pp, rr, ff, ss in zip(labels, p, r, f, s)
    }

def make_model():
    return RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )

def probabilities_to_argmax(model, probabilities):
    classes = np.array(model.classes_, dtype=object)
    return classes[np.argmax(probabilities, axis=1)]

def probabilities_to_n1_rescue(model, probabilities, threshold):
    classes = np.array(model.classes_, dtype=object)
    n1_idx = int(np.where(classes == "N1")[0][0])
    n1_prob = probabilities[:, n1_idx]
    pred = np.empty(len(probabilities), dtype=object)

    rescue = n1_prob >= threshold
    pred[rescue] = "N1"

    non_n1_idx = [i for i, c in enumerate(classes) if c != "N1"]

    if (~rescue).sum() > 0:
        non_n1_probs = probabilities[~rescue][:, non_n1_idx]
        rel = np.argmax(non_n1_probs, axis=1)
        model_idx = np.array(non_n1_idx)[rel]
        pred[~rescue] = classes[model_idx]

    return pred

def select_threshold_inner_loso(outer_train_raw, outer_train_subjects):
    records = {t: [] for t in threshold_candidates}

    for inner_test_subject in outer_train_subjects:
        inner_train_raw = outer_train_raw[outer_train_raw["subject"] != inner_test_subject].copy()
        inner_test_raw = outer_train_raw[outer_train_raw["subject"] == inner_test_subject].copy()

        means, stds = calculate_training_statistics(inner_train_raw, feature_columns)
        inner_train = normalize_with_fixed_statistics(inner_train_raw, feature_columns, means, stds)
        inner_test = normalize_with_fixed_statistics(inner_test_raw, feature_columns, means, stds)

        X_train = inner_train[feature_columns]
        y_train = inner_train["label"]
        X_test = inner_test[feature_columns]
        y_test = inner_test["label"]

        model = make_model()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)

        metadata = inner_test_raw[["subject", "recording", "epoch", "start_sec", "label"]].copy()

        for threshold in threshold_candidates:
            raw_pred = probabilities_to_n1_rescue(model, probs, threshold)
            smoothed = smooth_predictions_by_recording(metadata, raw_pred)
            overall = overall_metrics(y_test, smoothed)
            per_class = per_class_metrics(y_test, smoothed, stage_order)
            min_class_f1 = min(per_class[s]["f1"] for s in stage_order)

            records[threshold].append(
                {
                    "subject": inner_test_subject,
                    "macro_f1": overall["macro_f1"],
                    "n1_f1": per_class["N1"]["f1"],
                    "min_class_f1": min_class_f1,
                }
            )

    rows = []
    for threshold, recs in records.items():
        r = pd.DataFrame(recs)
        rows.append(
            {
                "threshold": threshold,
                "mean_macro_f1": float(r["macro_f1"].mean()),
                "mean_n1_f1": float(r["n1_f1"].mean()),
                "mean_min_class_f1": float(r["min_class_f1"].mean()),
                "std_macro_f1": float(r["macro_f1"].std(ddof=0)),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["mean_macro_f1", "mean_n1_f1", "mean_min_class_f1"],
        ascending=False,
    ).reset_index(drop=True)

    return float(summary.iloc[0]["threshold"]), summary

outer_rows = []
inner_rows = []
epoch_frames = []

print("\n" + "=" * 92)
print("STARTING NESTED LOSO CALIBRATION")
print("=" * 92)

for outer_idx, outer_test_subject in enumerate(subjects, start=1):
    print("\n" + "#" * 92)
    print(f"OUTER FOLD {outer_idx}/{len(subjects)} - TEST: {outer_test_subject}")
    print("#" * 92)

    outer_train_raw = df_raw[df_raw["subject"] != outer_test_subject].copy()
    outer_test_raw = df_raw[df_raw["subject"] == outer_test_subject].copy()
    outer_train_subjects = sorted(outer_train_raw["subject"].unique())

    inner_start = time.perf_counter()
    selected_threshold, threshold_summary = select_threshold_inner_loso(
        outer_train_raw, outer_train_subjects
    )
    inner_time = time.perf_counter() - inner_start

    print("Selected threshold:", selected_threshold)

    for _, row in threshold_summary.iterrows():
        inner_rows.append(
            {
                "outer_test_subject": outer_test_subject,
                "threshold": float(row["threshold"]),
                "mean_macro_f1": float(row["mean_macro_f1"]),
                "mean_n1_f1": float(row["mean_n1_f1"]),
                "mean_min_class_f1": float(row["mean_min_class_f1"]),
                "std_macro_f1": float(row["std_macro_f1"]),
                "selected": bool(float(row["threshold"]) == selected_threshold),
            }
        )

    means, stds = calculate_training_statistics(outer_train_raw, feature_columns)
    outer_train = normalize_with_fixed_statistics(outer_train_raw, feature_columns, means, stds)
    outer_test = normalize_with_fixed_statistics(outer_test_raw, feature_columns, means, stds)

    X_train = outer_train[feature_columns]
    y_train = outer_train["label"]
    X_test = outer_test[feature_columns]
    y_test = outer_test["label"]

    model = make_model()
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start
    probs = model.predict_proba(X_test)

    metadata = outer_test_raw[["subject", "recording", "epoch", "start_sec", "label"]].copy()

    baseline_raw = probabilities_to_argmax(model, probs)
    baseline_smoothed = smooth_predictions_by_recording(metadata, baseline_raw)

    calibrated_raw = probabilities_to_n1_rescue(model, probs, selected_threshold)
    calibrated_smoothed = smooth_predictions_by_recording(metadata, calibrated_raw)

    baseline_overall = overall_metrics(y_test, baseline_smoothed)
    calibrated_overall = overall_metrics(y_test, calibrated_smoothed)
    baseline_class = per_class_metrics(y_test, baseline_smoothed, stage_order)
    calibrated_class = per_class_metrics(y_test, calibrated_smoothed, stage_order)

    row = {
        "test_subject": outer_test_subject,
        "selected_threshold": selected_threshold,
        "training_time_sec": training_time,
        "inner_selection_time_sec": inner_time,
        "baseline_macro_f1": baseline_overall["macro_f1"],
        "calibrated_macro_f1": calibrated_overall["macro_f1"],
        "macro_f1_delta": calibrated_overall["macro_f1"] - baseline_overall["macro_f1"],
        "baseline_accuracy": baseline_overall["accuracy"],
        "calibrated_accuracy": calibrated_overall["accuracy"],
        "baseline_balanced_accuracy": baseline_overall["balanced_accuracy"],
        "calibrated_balanced_accuracy": calibrated_overall["balanced_accuracy"],
    }

    for stage in stage_order:
        row[f"baseline_{stage}_f1"] = baseline_class[stage]["f1"]
        row[f"calibrated_{stage}_f1"] = calibrated_class[stage]["f1"]
        row[f"delta_{stage}_f1"] = calibrated_class[stage]["f1"] - baseline_class[stage]["f1"]

    outer_rows.append(row)

    classes = np.array(model.classes_, dtype=object)
    n1_idx = int(np.where(classes == "N1")[0][0])

    epoch_df = metadata.copy()
    epoch_df["selected_threshold"] = selected_threshold
    epoch_df["n1_probability"] = probs[:, n1_idx]
    epoch_df["baseline_prediction"] = baseline_smoothed
    epoch_df["calibrated_prediction"] = calibrated_smoothed
    epoch_df["prediction_changed"] = baseline_smoothed != calibrated_smoothed
    epoch_frames.append(epoch_df)

    print("Baseline Macro F1:", round(baseline_overall["macro_f1"], 4))
    print("Calibrated Macro F1:", round(calibrated_overall["macro_f1"], 4))
    print("Delta:", round(row["macro_f1_delta"], 4))
    print("Baseline N1 F1:", round(baseline_class["N1"]["f1"], 4))
    print("Calibrated N1 F1:", round(calibrated_class["N1"]["f1"], 4))

fold_results_df = pd.DataFrame(outer_rows)
inner_threshold_df = pd.DataFrame(inner_rows)
epoch_predictions_df = pd.concat(epoch_frames, ignore_index=True)

mean_baseline_macro = float(fold_results_df["baseline_macro_f1"].mean())
mean_calibrated_macro = float(fold_results_df["calibrated_macro_f1"].mean())
mean_macro_delta = mean_calibrated_macro - mean_baseline_macro

mean_baseline_n1 = float(fold_results_df["baseline_N1_f1"].mean())
mean_calibrated_n1 = float(fold_results_df["calibrated_N1_f1"].mean())
mean_n1_delta = mean_calibrated_n1 - mean_baseline_n1

pooled_baseline = overall_metrics(
    epoch_predictions_df["label"],
    epoch_predictions_df["baseline_prediction"],
)
pooled_calibrated = overall_metrics(
    epoch_predictions_df["label"],
    epoch_predictions_df["calibrated_prediction"],
)

pooled_baseline_class = per_class_metrics(
    epoch_predictions_df["label"],
    epoch_predictions_df["baseline_prediction"],
    stage_order,
)
pooled_calibrated_class = per_class_metrics(
    epoch_predictions_df["label"],
    epoch_predictions_df["calibrated_prediction"],
    stage_order,
)

print("\n" + "=" * 92)
print("STAGE 38 OUTER-FOLD RESULTS")
print("=" * 92)

print(
    fold_results_df[
        [
            "test_subject",
            "selected_threshold",
            "baseline_macro_f1",
            "calibrated_macro_f1",
            "macro_f1_delta",
            "baseline_N1_f1",
            "calibrated_N1_f1",
            "delta_N1_f1",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

class_rows = []
for stage in stage_order:
    class_rows.append(
        {
            "stage": stage,
            "baseline_f1": pooled_baseline_class[stage]["f1"],
            "calibrated_f1": pooled_calibrated_class[stage]["f1"],
            "delta_f1": pooled_calibrated_class[stage]["f1"] - pooled_baseline_class[stage]["f1"],
        }
    )

class_comparison_df = pd.DataFrame(class_rows)

print("\n" + "=" * 92)
print("POOLED PER-CLASS F1 COMPARISON")
print("=" * 92)
print(class_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

threshold_counts_df = (
    fold_results_df["selected_threshold"]
    .value_counts()
    .sort_index()
    .rename_axis("threshold")
    .reset_index(name="outer_fold_count")
)

changed_epochs = int(epoch_predictions_df["prediction_changed"].sum())
changed_percent = changed_epochs / len(epoch_predictions_df) * 100

print("\n" + "=" * 92)
print("STAGE 38 SUMMARY")
print("=" * 92)

print("Mean baseline LOSO Macro F1:", round(mean_baseline_macro, 4))
print("Mean calibrated LOSO Macro F1:", round(mean_calibrated_macro, 4))
print("Mean Macro F1 delta:", round(mean_macro_delta, 4))
print()
print("Mean baseline N1 F1:", round(mean_baseline_n1, 4))
print("Mean calibrated N1 F1:", round(mean_calibrated_n1, 4))
print("Mean N1 F1 delta:", round(mean_n1_delta, 4))
print()
print("Pooled baseline Macro F1:", round(pooled_baseline["macro_f1"], 4))
print("Pooled calibrated Macro F1:", round(pooled_calibrated["macro_f1"], 4))
print()
print("Prediction changes:", changed_epochs, "/", len(epoch_predictions_df), f"({changed_percent:.2f}%)")
print("\nSelected threshold distribution:")
print(threshold_counts_df.to_string(index=False))

print("\nInterpretation:")
if mean_macro_delta > 0.01 and mean_n1_delta > 0.03:
    print("PROMISING: threshold calibration improves overall LOSO Macro F1 and N1.")
elif mean_macro_delta > 0:
    print("MARGINAL IMPROVEMENT: threshold calibration gives a small cross-subject gain.")
else:
    print("NO OVERALL BENEFIT: move next to temporal decoding rather than static threshold rescue.")

fold_results_path = results_dir / "stage38_outer_fold_results.csv"
inner_threshold_path = results_dir / "stage38_inner_threshold_search.csv"
class_comparison_path = results_dir / "stage38_per_class_comparison.csv"
epoch_predictions_path = results_dir / "stage38_epoch_predictions.csv"
threshold_counts_path = results_dir / "stage38_threshold_distribution.csv"
summary_path = results_dir / "stage38_summary.csv"

fold_results_df.to_csv(fold_results_path, index=False)
inner_threshold_df.to_csv(inner_threshold_path, index=False)
class_comparison_df.to_csv(class_comparison_path, index=False)
epoch_predictions_df.to_csv(epoch_predictions_path, index=False)
threshold_counts_df.to_csv(threshold_counts_path, index=False)

pd.DataFrame(
    [
        {
            "mean_baseline_loso_macro_f1": mean_baseline_macro,
            "mean_calibrated_loso_macro_f1": mean_calibrated_macro,
            "mean_macro_f1_delta": mean_macro_delta,
            "mean_baseline_n1_f1": mean_baseline_n1,
            "mean_calibrated_n1_f1": mean_calibrated_n1,
            "mean_n1_f1_delta": mean_n1_delta,
            "pooled_baseline_macro_f1": pooled_baseline["macro_f1"],
            "pooled_calibrated_macro_f1": pooled_calibrated["macro_f1"],
            "changed_epoch_count": changed_epochs,
            "changed_epoch_percent": changed_percent,
        }
    ]
).to_csv(summary_path, index=False)

print("\nSaved outputs:")
for p in [
    fold_results_path,
    inner_threshold_path,
    class_comparison_path,
    epoch_predictions_path,
    threshold_counts_path,
    summary_path,
]:
    print(p)

print("\n" + "=" * 92)
print("STAGE 38 COMPLETE")
print("=" * 92)
