from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "outputs" / "multi_subject_temporal_context.csv"
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

print("=" * 86)
print("STAGE 36 - N1 FEATURE DISCRIMINABILITY AUDIT")
print("=" * 86)

df = pd.read_csv(data_path)

metadata_columns = ["subject", "recording", "epoch", "start_sec", "label"]
feature_columns = [c for c in df.columns if c not in metadata_columns]
subjects = sorted(df["subject"].unique())

print("\nSubjects:", subjects)
print("Total epochs:", len(df))
print("Feature count:", len(feature_columns))

nan_count = int(df[feature_columns].isna().sum().sum())
inf_count = int(np.isinf(df[feature_columns].to_numpy()).sum())

print("NaN count:", nan_count)
print("Inf count:", inf_count)

if nan_count != 0 or inf_count != 0:
    raise ValueError("Feature table contains NaN or Inf.")

target_stage = "N1"
comparison_stages = [s for s in ["Wake", "N2", "REM"] if s in set(df["label"].unique())]

print("\nTarget stage:", target_stage)
print("Comparisons:", comparison_stages)


def cohens_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if len(x1) < 2 or len(x2) < 2:
        return 0.0

    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)
    denominator = len(x1) + len(x2) - 2

    if denominator <= 0:
        return 0.0

    pooled_var = (
        ((len(x1) - 1) * var1 + (len(x2) - 1) * var2)
        / denominator
    )

    if np.isnan(pooled_var) or pooled_var <= 1e-12:
        return 0.0

    return float((np.mean(x1) - np.mean(x2)) / math.sqrt(pooled_var))


def symmetric_auc(y_true, values):
    if len(np.unique(values)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(y_true, values)
    except ValueError:
        return 0.5

    return float(max(auc, 1.0 - auc))


def histogram_overlap(x1, x2, bins=40):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    combined = np.concatenate([x1, x2])

    min_value = float(np.min(combined))
    max_value = float(np.max(combined))

    if math.isclose(min_value, max_value):
        return 1.0

    edges = np.linspace(min_value, max_value, bins + 1)

    hist1, _ = np.histogram(x1, bins=edges, density=True)
    hist2, _ = np.histogram(x2, bins=edges, density=True)

    overlap = np.sum(np.minimum(hist1, hist2) * np.diff(edges))

    return float(np.clip(overlap, 0.0, 1.0))


# ------------------------------------------------------------
# Pairwise univariate feature audit
# ------------------------------------------------------------

feature_audit_rows = []

for comparison_stage in comparison_stages:
    print(f"\nAnalyzing N1 vs {comparison_stage} ...")

    pair_df = df[df["label"].isin([target_stage, comparison_stage])].copy()
    y_binary = (pair_df["label"] == target_stage).astype(int)

    for feature in feature_columns:
        n1_values = pair_df[pair_df["label"] == target_stage][feature].to_numpy()
        cmp_values = pair_df[pair_df["label"] == comparison_stage][feature].to_numpy()

        d = cohens_d(n1_values, cmp_values)
        auc = symmetric_auc(y_binary, pair_df[feature].to_numpy())
        overlap = histogram_overlap(n1_values, cmp_values)

        auc_component = (auc - 0.5) / 0.5
        d_component = min(abs(d) / 2.0, 1.0)
        overlap_component = 1.0 - overlap

        score = (
            0.50 * auc_component
            + 0.30 * d_component
            + 0.20 * overlap_component
        )

        feature_audit_rows.append(
            {
                "target_stage": target_stage,
                "comparison_stage": comparison_stage,
                "feature": feature,
                "n1_count": len(n1_values),
                "comparison_count": len(cmp_values),
                "n1_mean": float(np.mean(n1_values)),
                "comparison_mean": float(np.mean(cmp_values)),
                "cohens_d": d,
                "abs_cohens_d": abs(d),
                "symmetric_auc": auc,
                "overlap_coefficient": overlap,
                "discriminability_score": score,
            }
        )

feature_audit_df = pd.DataFrame(feature_audit_rows)

feature_audit_df["pair_rank"] = (
    feature_audit_df
    .groupby("comparison_stage")["discriminability_score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

for comparison_stage in comparison_stages:
    subset = (
        feature_audit_df[
            feature_audit_df["comparison_stage"] == comparison_stage
        ]
        .sort_values("discriminability_score", ascending=False)
        .head(10)
    )

    print("\n" + "=" * 86)
    print(f"TOP FEATURES: N1 vs {comparison_stage}")
    print("=" * 86)
    print(
        subset[
            [
                "pair_rank",
                "feature",
                "symmetric_auc",
                "cohens_d",
                "overlap_coefficient",
                "discriminability_score",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


# ------------------------------------------------------------
# Overall feature ranking
# ------------------------------------------------------------

summary_rows = []

for feature in feature_columns:
    subset = feature_audit_df[feature_audit_df["feature"] == feature]

    summary_rows.append(
        {
            "feature": feature,
            "mean_symmetric_auc": float(subset["symmetric_auc"].mean()),
            "min_symmetric_auc": float(subset["symmetric_auc"].min()),
            "mean_abs_cohens_d": float(subset["abs_cohens_d"].mean()),
            "mean_overlap_coefficient": float(subset["overlap_coefficient"].mean()),
            "mean_discriminability_score": float(subset["discriminability_score"].mean()),
            "best_separated_pair": str(
                subset.sort_values("discriminability_score", ascending=False)
                .iloc[0]["comparison_stage"]
            ),
            "worst_separated_pair": str(
                subset.sort_values("discriminability_score", ascending=True)
                .iloc[0]["comparison_stage"]
            ),
        }
    )

feature_summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values("mean_discriminability_score", ascending=False)
    .reset_index(drop=True)
)

feature_summary_df["overall_rank"] = np.arange(len(feature_summary_df)) + 1

print("\n" + "=" * 86)
print("OVERALL N1 FEATURE RANKING")
print("=" * 86)
print(
    feature_summary_df[
        [
            "overall_rank",
            "feature",
            "mean_symmetric_auc",
            "min_symmetric_auc",
            "mean_abs_cohens_d",
            "mean_overlap_coefficient",
            "mean_discriminability_score",
            "best_separated_pair",
            "worst_separated_pair",
        ]
    ]
    .head(20)
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)


# ------------------------------------------------------------
# Pair difficulty summary
# ------------------------------------------------------------

pair_summary_rows = []

for comparison_stage in comparison_stages:
    subset = feature_audit_df[
        feature_audit_df["comparison_stage"] == comparison_stage
    ]
    top10 = subset.sort_values("discriminability_score", ascending=False).head(10)

    pair_summary_rows.append(
        {
            "comparison_stage": comparison_stage,
            "best_feature": top10.iloc[0]["feature"],
            "best_feature_auc": float(top10.iloc[0]["symmetric_auc"]),
            "best_feature_abs_d": float(abs(top10.iloc[0]["cohens_d"])),
            "best_feature_overlap": float(top10.iloc[0]["overlap_coefficient"]),
            "mean_top10_auc": float(top10["symmetric_auc"].mean()),
            "mean_top10_abs_d": float(top10["abs_cohens_d"].mean()),
            "mean_top10_overlap": float(top10["overlap_coefficient"].mean()),
            "mean_top10_score": float(top10["discriminability_score"].mean()),
        }
    )

pair_summary_df = (
    pd.DataFrame(pair_summary_rows)
    .sort_values("mean_top10_score", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 86)
print("PAIRWISE DIFFICULTY SUMMARY")
print("=" * 86)
print(pair_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ------------------------------------------------------------
# Pairwise RF LOSO separability diagnostic
# ------------------------------------------------------------

rf_rows = []

for comparison_stage in comparison_stages:
    pair_df = df[df["label"].isin([target_stage, comparison_stage])].copy()

    for test_subject in subjects:
        train_df = pair_df[pair_df["subject"] != test_subject].copy()
        test_df = pair_df[pair_df["subject"] == test_subject].copy()

        if len(test_df) == 0:
            continue

        y_train = (train_df["label"] == target_stage).astype(int)
        y_test = (test_df["label"] == target_stage).astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        means = train_df[feature_columns].mean()
        stds = train_df[feature_columns].std().replace(0, 1.0).fillna(1.0)

        X_train = ((train_df[feature_columns] - means) / (stds + 1e-8))
        X_test = ((test_df[feature_columns] - means) / (stds + 1e-8))

        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        )

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        rf_rows.append(
            {
                "comparison_stage": comparison_stage,
                "test_subject": test_subject,
                "auc": roc_auc_score(y_test, proba),
                "accuracy": float(np.mean(pred == y_test.to_numpy())),
                "n_test": len(test_df),
            }
        )

pairwise_rf_df = pd.DataFrame(rf_rows)

rf_summary_rows = []

for comparison_stage in comparison_stages:
    subset = pairwise_rf_df[
        pairwise_rf_df["comparison_stage"] == comparison_stage
    ]

    if len(subset) == 0:
        continue

    rf_summary_rows.append(
        {
            "comparison_stage": comparison_stage,
            "mean_loso_auc": float(subset["auc"].mean()),
            "std_loso_auc": float(subset["auc"].std(ddof=0)),
            "min_loso_auc": float(subset["auc"].min()),
            "max_loso_auc": float(subset["auc"].max()),
            "mean_accuracy": float(subset["accuracy"].mean()),
        }
    )

pairwise_rf_summary_df = (
    pd.DataFrame(rf_summary_rows)
    .sort_values("mean_loso_auc", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 86)
print("PAIRWISE RF LOSO SEPARABILITY")
print("=" * 86)
print(
    pairwise_rf_summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

hardest_pair_univariate = str(
    pair_summary_df.sort_values("mean_top10_score", ascending=True)
    .iloc[0]["comparison_stage"]
)

hardest_pair_rf_row = (
    pairwise_rf_summary_df
    .sort_values("mean_loso_auc", ascending=True)
    .iloc[0]
)

hardest_pair_rf = str(hardest_pair_rf_row["comparison_stage"])
hardest_pair_rf_auc = float(hardest_pair_rf_row["mean_loso_auc"])

best_feature = str(feature_summary_df.iloc[0]["feature"])
best_feature_auc = float(feature_summary_df.iloc[0]["mean_symmetric_auc"])
best_feature_score = float(feature_summary_df.iloc[0]["mean_discriminability_score"])

print("\n" + "=" * 86)
print("STAGE 36 SUMMARY")
print("=" * 86)

print("\nBest overall feature:", best_feature)
print("Mean symmetric AUC:", round(best_feature_auc, 4))
print("Mean discriminability score:", round(best_feature_score, 4))

print(
    "\nHardest N1 pair by univariate separation:",
    f"N1 vs {hardest_pair_univariate}",
)

print(
    "Hardest N1 pair by pairwise RF LOSO:",
    f"N1 vs {hardest_pair_rf}",
)

print(
    "Hardest-pair RF mean LOSO AUC:",
    round(hardest_pair_rf_auc, 4),
)

print("\nDiagnostic interpretation:")

if hardest_pair_rf_auc < 0.65:
    print("WEAK SEPARABILITY.")
    print("Stage 37 should add targeted physiological features.")
elif hardest_pair_rf_auc < 0.75:
    print("MODERATE SEPARABILITY.")
    print("Stage 37 should test targeted feature augmentation.")
else:
    print("STRONG PAIRWISE SIGNAL.")
    print(
        "The N1 bottleneck may be driven more by multiclass "
        "decision boundaries or temporal context."
    )


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

feature_audit_path = results_dir / "stage36_pairwise_feature_audit.csv"
feature_summary_path = results_dir / "stage36_feature_ranking.csv"
pair_summary_path = results_dir / "stage36_pairwise_difficulty.csv"
pairwise_rf_path = results_dir / "stage36_pairwise_rf_loso.csv"
pairwise_rf_summary_path = results_dir / "stage36_pairwise_rf_summary.csv"
summary_path = results_dir / "stage36_summary.csv"

feature_audit_df.to_csv(feature_audit_path, index=False)
feature_summary_df.to_csv(feature_summary_path, index=False)
pair_summary_df.to_csv(pair_summary_path, index=False)
pairwise_rf_df.to_csv(pairwise_rf_path, index=False)
pairwise_rf_summary_df.to_csv(pairwise_rf_summary_path, index=False)

pd.DataFrame(
    [
        {
            "target_stage": target_stage,
            "feature_count": len(feature_columns),
            "best_overall_feature": best_feature,
            "best_overall_feature_mean_auc": best_feature_auc,
            "best_overall_feature_score": best_feature_score,
            "hardest_pair_univariate": hardest_pair_univariate,
            "hardest_pair_rf": hardest_pair_rf,
            "hardest_pair_rf_mean_auc": hardest_pair_rf_auc,
        }
    ]
).to_csv(summary_path, index=False)

print("\nSaved feature audit to:")
print(feature_audit_path)

print("\nSaved feature ranking to:")
print(feature_summary_path)

print("\nSaved pair difficulty summary to:")
print(pair_summary_path)

print("\nSaved RF LOSO results to:")
print(pairwise_rf_path)

print("\nSaved RF summary to:")
print(pairwise_rf_summary_path)

print("\nSaved summary to:")
print(summary_path)

print("\n" + "=" * 86)
print("STAGE 36 COMPLETE")
print("=" * 86)
