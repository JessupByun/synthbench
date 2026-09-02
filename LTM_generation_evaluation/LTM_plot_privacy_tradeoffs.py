#!/usr/bin/env python3
"""
plot_privacy_tradeoff.py

Generate a 2×2 grid of scatterplots showing privacy (mean worst-case AUC)
versus fidelity/utility/diversity metrics, with privacy on the X-axis,
points colored by model, and a shared legend at the bottom.
Each subplot has its own X and Y axis labels.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === USER CONFIGURATION ===
INPUT_CSV   = 'LTM_evaluation/big_data_with_util_diversity.csv'
OUTPUT_PNG  = 'LTM_evaluation/privacy_tradeoff_grid_individual_labels.png'
# ==========================

# Define model order and assign colors
MODELS = [
    'SMOTE', 'CTGAN', 'TVAE', 'TabDiff',
    'GPT-4o-mini', 'TabPFNv2', 'LLaMA 3.3 70B'
]
CMAP = plt.get_cmap('tab10')
COLORS = {m: CMAP(i) for i, m in enumerate(MODELS)}

# === Load & preprocess data ===
df = pd.read_csv(INPUT_CSV)
df = df[df['Reference'] == True]

# Standardize model names (catch substrings so suffixes get normalized)
def normalize_model(raw: str) -> str:
    m = raw.lower()
    if 'gpt-4o-mini' in m:
        return 'GPT-4o-mini'
    if 'ctgan' in m:
        return 'CTGAN'
    if 'smote' in m:
        return 'SMOTE'
    if 'tabdiff' in m:
        return 'TabDiff'
    if 'llama' in m:
        return 'LLaMA 3.3 70B'
    if 'tabpfn' in m:
        return 'TabPFNv2'
    if 'tvae' in m:
        return 'TVAE'
    return raw

df['Model'] = df['Model'].apply(normalize_model)

# === Compute privacy (X-axis) per model ===
privacy_split = (
    df.groupby(['Model','Dataset','TrainingSize','Seed'])['AUC_Score']
      .max()
      .reset_index(name='Max_AUC')
)
privacy_mean = privacy_split.groupby('Model')['Max_AUC'].mean()

# === Prepare data-quality metrics (Y-axis) ===
split_metrics = df.drop_duplicates(subset=['Model','Dataset','TrainingSize','Seed'])
metrics = {
    'Correlation Similarity': split_metrics.groupby('Model')['avg_corr_similarity'].mean(),
    'Shape Similarity':       split_metrics.groupby('Model')['avg_shape'].mean(),
    'Utility AUC':            split_metrics.groupby('Model')['avg_ROC_AUC'].mean(),
    'Beta Recall':            split_metrics.groupby('Model')['delta_coverage_beta_naive'].mean(),
}

# === Plot grid with individual labels ===
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)
axes = axes.flatten()

for ax, (metric_name, y_vals) in zip(axes, metrics.items()):
    # Plot each model's point
    for m in MODELS:
        if m in y_vals.index:
            ax.scatter(
                privacy_mean[m],  # X: privacy
                y_vals[m],        # Y: metric
                color=COLORS[m],
                edgecolor='k', s=80, zorder=3
            )
    # Title and axis labels per subplot
    ax.set_title(f'Privacy vs. {metric_name}', fontsize=12)
    ax.set_xlabel('Average Max AUC (Privacy)', fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

# Shared legend at bottom with compact spacing
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS[m], markeredgecolor='k', markersize=8)
    for m in MODELS
]
fig.legend(
    handles, MODELS,
    loc='lower center', ncol=len(MODELS),
    frameon=False, fontsize=10,
    labelspacing=0.2, columnspacing=0.5, handletextpad=0.5
)

# Adjust layout to fit legend and spacing
fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.25)
fig.savefig(OUTPUT_PNG, dpi=300)
plt.close(fig)

print(f"✅ Saved grid with individual labels: {OUTPUT_PNG}")
