#!/usr/bin/env python3
"""
Post-process the merged big-data CSV to produce the main privacy score table.

For each split (dataset, size, seed), we select the *max* attacker metric
(worst-case leakage), then average those max‐values across all splits
(35 datasets × 3 seeds × 3 sizes = 315 values per model).

USAGE:
  1. Edit the USER CONFIGURATION section in `main()`.
  2. Run:
       python postprocess_privacy.py
"""
import pandas as pd
import numpy as np
from tabulate import tabulate

def main():
    # ===== USER CONFIGURATION =====
    INPUT_CSV        = 'LTM_evaluation/big_data_public.csv'      # your merged table
    OUTPUT_CSV       = 'LTM_evaluation/privacy_score_table.csv'
    FILTER_REFERENCE = True                       # only include Reference==True runs?
    METRICS = [
        'AUC_Score',
        'TPR@FPR=0',
        'TPR@FPR=0.001',
        'TPR@FPR=0.01',
        'TPR@FPR=0.1'
    ]
    # ==============================

    # 1) Load
    df = pd.read_csv(INPUT_CSV)

    # 2) Optionally filter out the no-ref runs
    if FILTER_REFERENCE:
        df = df[df['Reference'] == True]

    # 3) For each split: collapse all attackers to the single worst-case metric
    group_cols = ['Model', 'Dataset', 'TrainingSize', 'Seed']
    df_max = df.groupby(group_cols)[METRICS].max().reset_index()

    # 4) Aggregate across *all* splits (315 total) by model
    agg = df_max.groupby('Model')[METRICS].agg(['mean','std'])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]

    # 5) Format "mean (std)" 
    summary = pd.DataFrame(index=agg.index)
    for metric in METRICS:
        summary[metric] = (
            agg[f"{metric}_mean"].round(3).astype(str)
            + ' ('
            + agg[f"{metric}_std"].round(3).astype(str)
            + ')'
        )

    # 6) Output
    summary.to_csv(OUTPUT_CSV)
    print(f"Saved privacy score table to {OUTPUT_CSV}\n")

    print("Privacy Score Table (all sizes, n=32,64,128):\n")
    print(tabulate(summary.reset_index(),
                   headers='keys',
                   tablefmt='github',
                   showindex=False))

if __name__ == '__main__':
    main()
