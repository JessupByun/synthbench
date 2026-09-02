#!/usr/bin/env python3
"""
select_ablation_datasets.py

Pick the top-K datasets for ablation, based on worst-case AUC,
for a given generator at a given subset size.

Usage:
  Edit USER CONFIGURATION below, then run:
    python select_ablation_datasets.py
"""
import pandas as pd

# ===== USER CONFIGURATION =====
INPUT_CSV  = 'LTM_evaluation/big_data_with_util_diversity.csv'
MODEL_NAME = 'llama'   # match exactly how 'llama' appears in your Model column
TRAIN_SIZE = 32        # subset size to focus on
TOP_K      = 10        # number of datasets to select
# ==============================

def main():
    df = pd.read_csv(INPUT_CSV)
    # Case‐insensitive match on Model
    df = df[df['Model'].str.lower() == MODEL_NAME.lower()]
    df = df[df['TrainingSize'] == TRAIN_SIZE]

    # Collapse each split (Dataset + Seed) to its worst-case AUC
    privacy_split = (
        df.groupby(['Dataset','Seed'])['AUC_Score']
          .max()
          .reset_index(name='Max_AUC')
    )

    # Compute mean & std of those Max_AUCs per dataset
    agg = privacy_split.groupby('Dataset')['Max_AUC'] \
               .agg(['mean','std']) \
               .reset_index() \
               .rename(columns={'mean':'Mean_Max_AUC','std':'Std_Max_AUC'})

    # Sort descending and take top-K
    top = agg.sort_values('Mean_Max_AUC', ascending=False).head(TOP_K)

    # Output
    print(f"Top {TOP_K} datasets for {MODEL_NAME} at n={TRAIN_SIZE}:")
    print(top.to_string(index=False))

    out_csv = f"LTM_evaluation/ablation_datasets_{MODEL_NAME}_n{TRAIN_SIZE}.csv"
    top.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

if __name__ == '__main__':
    main()
