#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
INPUT_CSV        = 'LTM_evaluation/big_data_public.csv'
OUTPUT_PNG       = 'LTM_evaluation/barplot_across_sizes.png'
FILTER_REFERENCE = True
METRIC           = 'AUC_Score'
GROUP_COLS       = ['Model', 'Dataset', 'TrainingSize', 'Seed']

# === Helpers ===
def rename_model(raw: str) -> str:
    m = raw.lower()
    if 'ctgan' in m: return 'CTGAN'
    if 'smote' in m: return 'SMOTE'
    if 'tabdiff' in m: return 'TabDiff'
    if 'llama' in m: return 'LLaMA 3.3 70B'
    if 'gpt-4o-mini' in m: return 'GPT-4o-mini'
    if 'tabpfn' in m: return 'TabPFNv2'
    if 'tvae' in m: return 'TVAE'
    return raw

# === Load & Preprocess ===
df = pd.read_csv(INPUT_CSV)
if FILTER_REFERENCE:
    df = df[df['Reference'] == True]

df['Model'] = df['Model'].apply(rename_model)

# Max AUC per group
df_max = df.groupby(GROUP_COLS)[METRIC].max().reset_index()
agg    = df_max.groupby(['Model', 'TrainingSize'])[METRIC].mean().reset_index()

# Force model order
desired_order = ['SMOTE','CTGAN','TVAE','TabDiff','GPT-4o-mini','TabPFNv2','LLaMA 3.3 70B']
agg['Model']  = pd.Categorical(agg['Model'], categories=desired_order, ordered=True)
agg = agg.sort_values('Model')

# Pivot for plotting
pivot = agg.pivot(index='Model', columns='TrainingSize', values=METRIC)
sizes = sorted(pivot.columns.tolist())
models= pivot.index.tolist()

# === Plot ===
# 1) Reduce height: second arg in figsize
fig, ax = plt.subplots(figsize=(10, 4))  

# 2) Color setup
cmap = plt.get_cmap('Pastel1')
colors = [cmap(i) for i in range(len(sizes))]

x     = np.arange(len(models))
width = 0.2

for i, size in enumerate(sizes):
    ax.bar(
        x + (i - (len(sizes)-1)/2)*(width+0.02),
        pivot[size],
        width,
        label=f'n={size}',
        color=colors[i],
        edgecolor='k'
    )
    for j, val in enumerate(pivot[size]):
        ax.text(
            x[j] + (i - (len(sizes)-1)/2)*(width+0.02),
            val + 0.002,           # tiny offset
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=6
        )

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='center', fontsize=10)
ax.set_ylabel('Average Max AUC', fontsize=12)
ax.tick_params(axis='y', labelsize=10)

# 3) Manually tighten y-range around your data
min_val, max_val = pivot.values.min(), pivot.values.max()
padding = (max_val - min_val) * 0.1
ax.set_ylim(min_val - padding, max_val + padding)

ax.legend(
    title='Subset Size',
    fontsize=10,
    title_fontsize=10,
    loc='upper left',
    bbox_to_anchor=(1,1)
)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()
