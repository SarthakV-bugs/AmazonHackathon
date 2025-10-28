import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Create output directory for EDA results
output_dir = "eda_train_test_comparison"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("TRAIN VS TEST EDA COMPARISON")
print("=" * 70)

# -------------------------
# LOAD DATA
# -------------------------
print("\nLoading datasets...")

# UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS
train_file = "Final_npz/train_features.npz"  # UPDATE THIS
test_file = "Final_npz/test_features.npz"    # UPDATE THIS

train_data = np.load(train_file)
test_data = np.load(test_file)

train_sample_ids = train_data["sample_ids"]
train_features = train_data["features"]

test_sample_ids = test_data["sample_ids"]
test_features = test_data["features"]

print(f"\nTrain set: {train_features.shape[0]} samples, {train_features.shape[1]} features")
print(f"Test set:  {test_features.shape[0]} samples, {test_features.shape[1]} features")

# -------------------------
# CHECK FOR CRITICAL ISSUES
# -------------------------
print("\n" + "=" * 70)
print("CRITICAL CHECKS")
print("=" * 70)

issues_found = []

# 1. Dimension mismatch
if train_features.shape[1] != test_features.shape[1]:
    print(f"❌ CRITICAL: Feature dimension mismatch!")
    print(f"   Train: {train_features.shape[1]} features")
    print(f"   Test:  {test_features.shape[1]} features")
    print(f"   → Model will fail during prediction!")
    issues_found.append("Feature dimension mismatch")
else:
    print(f"✓ Feature dimensions match: {train_features.shape[1]}")

# 2. NaN/Inf check
train_has_nan = np.isnan(train_features).any()
train_has_inf = np.isinf(train_features).any()
test_has_nan = np.isnan(test_features).any()
test_has_inf = np.isinf(test_features).any()

if train_has_nan or test_has_nan:
    print(f"❌ WARNING: NaN values detected!")
    print(f"   Train NaN: {train_has_nan}, Test NaN: {test_has_nan}")
    issues_found.append("NaN values present")
else:
    print(f"✓ No NaN values")

if train_has_inf or test_has_inf:
    print(f"❌ WARNING: Infinite values detected!")
    print(f"   Train Inf: {train_has_inf}, Test Inf: {test_has_inf}")
    issues_found.append("Infinite values present")
else:
    print(f"✓ No infinite values")

# -------------------------
# BASIC STATISTICS COMPARISON
# -------------------------
print("\n" + "=" * 70)
print("STATISTICAL COMPARISON")
print("=" * 70)

stats_comparison = {
    "Metric": ["Mean", "Std", "Min", "Max", "Median"],
    "Train": [
        np.mean(train_features),
        np.std(train_features),
        np.min(train_features),
        np.max(train_features),
        np.median(train_features)
    ],
    "Test": [
        np.mean(test_features),
        np.std(test_features),
        np.min(test_features),
        np.max(test_features),
        np.median(test_features)
    ]
}

stats_df = pd.DataFrame(stats_comparison)
stats_df["Diff%"] = ((stats_df["Test"] - stats_df["Train"]) / (stats_df["Train"].abs() + 1e-10) * 100).round(2)
stats_df["Train"] = stats_df["Train"].round(4)
stats_df["Test"] = stats_df["Test"].round(4)

print(stats_df.to_string(index=False))

# Flag large differences
for idx, row in stats_df.iterrows():
    if abs(row["Diff%"]) > 20:
        print(f"⚠️  Large difference in {row['Metric']}: {row['Diff%']:+.2f}%")
        issues_found.append(f"Large {row['Metric']} difference ({row['Diff%']:+.2f}%)")

# -------------------------
# VARIANCE ANALYSIS
# -------------------------
print("\n" + "=" * 70)
print("VARIANCE ANALYSIS")
print("=" * 70)

train_var = np.var(train_features, axis=0)
test_var = np.var(test_features, axis=0)

print(f"Mean variance:")
print(f"  Train: {train_var.mean():.4f}")
print(f"  Test:  {test_var.mean():.4f}")
print(f"  Diff:  {((test_var.mean() - train_var.mean()) / train_var.mean() * 100):+.2f}%")

train_low_var = (train_var < 0.01).sum()
test_low_var = (test_var < 0.01).sum()

print(f"\nLow variance features (<0.01):")
print(f"  Train: {train_low_var} / {len(train_var)} ({train_low_var/len(train_var)*100:.1f}%)")
print(f"  Test:  {test_low_var} / {len(test_var)} ({test_low_var/len(test_var)*100:.1f}%)")

# Plot variance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Variance distribution
axes[0].hist(train_var, bins=50, alpha=0.6, label='Train', edgecolor='black')
axes[0].hist(test_var, bins=50, alpha=0.6, label='Test', edgecolor='black')
axes[0].set_xlabel('Variance')
axes[0].set_ylabel('Count')
axes[0].set_title('Variance Distribution')
axes[0].legend()
axes[0].set_yscale('log')

# Sorted variance
axes[1].plot(sorted(train_var, reverse=True), label='Train', alpha=0.8)
axes[1].plot(sorted(test_var, reverse=True), label='Test', alpha=0.8)
axes[1].set_xlabel('Feature Rank')
axes[1].set_ylabel('Variance')
axes[1].set_title('Sorted Feature Variances')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Scatter comparison
axes[2].scatter(train_var, test_var, alpha=0.3, s=10)
max_var = max(train_var.max(), test_var.max())
axes[2].plot([0, max_var], [0, max_var], 'r--', alpha=0.5, label='y=x')
axes[2].set_xlabel('Train Variance')
axes[2].set_ylabel('Test Variance')
axes[2].set_title('Train vs Test Variance Correlation')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/variance_comparison.png", dpi=150)
print(f"\n✓ Saved: {output_dir}/variance_comparison.png")
plt.close()

# -------------------------
# DISTRIBUTION SHIFT
# -------------------------
print("\n" + "=" * 70)
print("DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)

# Feature-wise mean difference
train_mean_per_feature = np.mean(train_features, axis=0)
test_mean_per_feature = np.mean(test_features, axis=0)

mean_diff = test_mean_per_feature - train_mean_per_feature
l1_shift = np.mean(np.abs(mean_diff))
l2_shift = np.sqrt(np.mean(mean_diff ** 2))

print(f"Distribution shift metrics:")
print(f"  L1 distance: {l1_shift:.6f}")
print(f"  L2 distance: {l2_shift:.6f}")

if l1_shift > 0.01:
    print(f"⚠️  WARNING: Significant distribution shift detected!")
    print(f"   → Test data may have different characteristics than train")
    issues_found.append(f"Distribution shift (L1={l1_shift:.6f})")
else:
    print(f"✓ Distributions are similar")

# Plot distribution shift
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(mean_diff, bins=50, edgecolor='black')
plt.axvline(0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Feature Mean Difference (Test - Train)')
plt.ylabel('Count')
plt.title('Distribution of Feature Mean Differences')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
top_shift_idx = np.argsort(np.abs(mean_diff))[-20:]
plt.barh(range(20), mean_diff[top_shift_idx])
plt.axvline(0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Mean Difference (Test - Train)')
plt.ylabel('Feature Index')
plt.title('Top 20 Features with Largest Mean Shift')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{output_dir}/distribution_shift.png", dpi=150)
print(f"✓ Saved: {output_dir}/distribution_shift.png")
plt.close()

# -------------------------
# PCA COMPARISON
# -------------------------
print("\n" + "=" * 70)
print("PCA COMPARISON")
print("=" * 70)

n_components = min(50, train_features.shape[1], test_features.shape[1])

pca_train = PCA(n_components=n_components)
pca_test = PCA(n_components=n_components)

pca_train.fit(train_features)
pca_test.fit(test_features)

train_cumvar = np.cumsum(pca_train.explained_variance_ratio_)
test_cumvar = np.cumsum(pca_test.explained_variance_ratio_)

print(f"Cumulative variance explained:")
print(f"  Components    Train     Test      Diff")
print(f"  {'='*45}")
for n in [5, 10, 20]:
    if n <= len(train_cumvar):
        train_val = train_cumvar[n-1] * 100
        test_val = test_cumvar[n-1] * 100
        diff = test_val - train_val
        print(f"  {n:2d}          {train_val:6.2f}%   {test_val:6.2f}%   {diff:+6.2f}%")

train_n95 = np.argmax(train_cumvar >= 0.95) + 1
test_n95 = np.argmax(test_cumvar >= 0.95) + 1

print(f"\nComponents for 95% variance:")
print(f"  Train: {train_n95}")
print(f"  Test:  {test_n95}")

if abs(train_n95 - test_n95) > 5:
    print(f"⚠️  WARNING: Different intrinsic dimensionality!")
    issues_found.append(f"Dimensionality mismatch (Train={train_n95}, Test={test_n95})")

# Plot PCA comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(min(20, len(pca_train.explained_variance_ratio_))),
           pca_train.explained_variance_ratio_[:20], alpha=0.7, label='Train')
axes[0].bar(range(min(20, len(pca_test.explained_variance_ratio_))),
           pca_test.explained_variance_ratio_[:20], alpha=0.7, label='Test')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Variance per Component (First 20)')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].plot(train_cumvar, marker='o', markersize=3, label='Train', linewidth=2)
axes[1].plot(test_cumvar, marker='s', markersize=3, label='Test', linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', alpha=0.3, label='95%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/pca_comparison.png", dpi=150)
print(f"✓ Saved: {output_dir}/pca_comparison.png")
plt.close()

# -------------------------
# FEATURE RANGE COMPARISON
# -------------------------
print("\n" + "=" * 70)
print("FEATURE RANGE COMPARISON")
print("=" * 70)

train_min_per_feat = np.min(train_features, axis=0)
train_max_per_feat = np.max(train_features, axis=0)
test_min_per_feat = np.min(test_features, axis=0)
test_max_per_feat = np.max(test_features, axis=0)

# Check if test is outside train range
out_of_range_min = (test_min_per_feat < train_min_per_feat).sum()
out_of_range_max = (test_max_per_feat > train_max_per_feat).sum()

print(f"Features with test values outside train range:")
print(f"  Below train min: {out_of_range_min} features")
print(f"  Above train max: {out_of_range_max} features")

if out_of_range_min > 0 or out_of_range_max > 0:
    print(f"⚠️  WARNING: Test set has out-of-range values!")
    print(f"   → Model may extrapolate poorly on these features")
    issues_found.append(f"Out-of-range values ({out_of_range_min + out_of_range_max} features)")

# -------------------------
# FINAL SUMMARY
# -------------------------
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

summary_data = {
    "Metric": [
        "Train Samples",
        "Test Samples",
        "Features",
        "Mean Diff%",
        "Std Diff%",
        "Distribution Shift (L1)",
        "PCA Components (95%)",
        "Low Variance Features",
        "Out-of-Range Features"
    ],
    "Value": [
        train_features.shape[0],
        test_features.shape[0],
        train_features.shape[1],
        f"{stats_df[stats_df['Metric']=='Mean']['Diff%'].values[0]:+.2f}%",
        f"{stats_df[stats_df['Metric']=='Std']['Diff%'].values[0]:+.2f}%",
        f"{l1_shift:.6f}",
        f"Train={train_n95}, Test={test_n95}",
        f"Train={train_low_var}, Test={test_low_var}",
        f"{out_of_range_min + out_of_range_max}"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
print(summary_df.to_string(index=False))

print(f"\n{'='*70}")
if len(issues_found) == 0:
    print("✓ NO CRITICAL ISSUES FOUND")
    print("  Train and test sets are well-aligned")
else:
    print(f"⚠️  {len(issues_found)} ISSUE(S) FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")

print(f"\n{'='*70}")
print("RECOMMENDATIONS FOR REDUCING SMAPE:")
print(f"{'='*70}")

recommendations = []

if train_low_var > 600:  # Based on your previous 639/644
    recommendations.append(
        "1. APPLY PCA: 99% of features have low variance\n"
        f"   → Reduce to {train_n95} components (captures 95% variance)\n"
        "   → Will speed up training 10-50x and reduce overfitting"
    )

if l1_shift > 0.01:
    recommendations.append(
        "2. ADDRESS DISTRIBUTION SHIFT:\n"
        "   → Apply feature standardization (StandardScaler fitted on train)\n"
        "   → Consider robust scaling if outliers present"
    )

if out_of_range_min + out_of_range_max > 10:
    recommendations.append(
        "3. HANDLE OUT-OF-RANGE VALUES:\n"
        "   → Clip test features to train min/max range\n"
        "   → Or use robust normalization techniques"
    )

if abs(train_n95 - test_n95) > 5:
    recommendations.append(
        "4. DIFFERENT INTRINSIC DIMENSIONALITY:\n"
        "   → Test set has different complexity than train\n"
        "   → May indicate train/test split issues or data collection differences"
    )

if len(recommendations) == 0:
    print("✓ Data looks good! Focus on model hyperparameter tuning.")
else:
    for rec in recommendations:
        print(f"\n{rec}")

print(f"\n{'='*70}")
print("EDA COMPLETE!")
print(f"All results saved to: {output_dir}/")
print(f"{'='*70}")
