#!/usr/bin/env python3
"""
augment_big_table.py

Augment your big_data_public.csv with two new columns per split:
  - delta_coverage_beta_naive (diversity)
  - avg_ROC_AUC               (utility)

Utility ROC AUC is computed by averaging all valid
'ROC AUC' values in TabularUtility_synth.csv (zeros included),
skipping only invalid/missing entries. Missing files/values → 0.0.

Usage:
  1. Update the USER CONFIGURATION block below.
  2. Run: python augment_big_table.py
"""
import os
import re
import pandas as pd
import numpy as np

# ===== USER CONFIGURATION =====
BIG_TABLE_CSV = 'LTM_evaluation/big_data_public.csv'       # path to your big table
UTILITY_ROOT  = 'LTM_evaluation/utility_diversity_scores' # root dir of split folders
OUTPUT_TARGET = 'LTM_evaluation/big_data_with_util_diversity.csv'  # output CSV path
# Only include these subset sizes:
VALID_SIZES   = {32, 64, 128}
# ==============================

def parse_folder_name(folder):
    """
    Parse: <dataset>--train--<size>-seed<seed>_<Model>_default_*
    Returns (dataset, size, seed, model) or (None, None, None, None).
    """
    m = re.match(r'^(.*?)--train--(\d+)-seed(\d+)_(.*?)_default', folder)
    if not m:
        return None, None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)

def collect_scores(root_dir):
    """
    Walk root_dir, for each split folder of VALID_SIZES:
      - Read delta_coverage_beta_naive
      - Loop through 'ROC AUC' rows to avg them
    Missing → 0.0. Returns DataFrame with one row per split.
    """
    records = []
    for dirpath, _, files in os.walk(root_dir):
        base = os.path.basename(dirpath)
        dataset, size, seed, model = parse_folder_name(base)
        if not dataset or size not in VALID_SIZES:
            continue

        # diversity
        delta_cov = 0.0
        div_csv = os.path.join(dirpath, 'AlphaPrecision_holdout_vs_synth.csv')
        if os.path.isfile(div_csv):
            try:
                df_div = pd.read_csv(div_csv)
                if 'delta_coverage_beta_naive' in df_div.columns:
                    delta_cov = float(df_div.at[0,'delta_coverage_beta_naive'])
            except:
                pass

        # utility
        avg_auc = 0.0
        util_csv = os.path.join(dirpath, 'TabularUtility_synth.csv')
        if os.path.isfile(util_csv):
            try:
                df_util = pd.read_csv(util_csv)
                if 'ROC AUC' in df_util.columns:
                    total = 0.0
                    count = 0
                    for v in df_util['ROC AUC']:
                        try:
                            val = float(v)
                        except:
                            continue
                        if np.isnan(val):
                            continue
                        total += val
                        count += 1
                    if count > 0:
                        avg_auc = total / count
            except:
                pass

        records.append({
            'Model': model,
            'Dataset': dataset,
            'TrainingSize': size,
            'Seed': seed,
            'delta_coverage_beta_naive': delta_cov,
            'avg_ROC_AUC': avg_auc
        })

    return pd.DataFrame.from_records(records)

def main():
    # 1) load big table
    big_df = pd.read_csv(BIG_TABLE_CSV)

    # 2) collect utility/diversity scores
    scores_df = collect_scores(UTILITY_ROOT)

    # 3) merge on split keys
    merged = big_df.merge(
        scores_df,
        on=['Model','Dataset','TrainingSize','Seed'],
        how='left'
    )

    # 4) fill any missing with 0.0
    merged['delta_coverage_beta_naive'] = merged['delta_coverage_beta_naive'].fillna(0.0)
    merged['avg_ROC_AUC']               = merged['avg_ROC_AUC'].fillna(0.0)

    # 5) write output
    out = OUTPUT_TARGET
    if os.path.isdir(OUTPUT_TARGET):
        base = os.path.basename(BIG_TABLE_CSV).replace('.csv','_augmented.csv')
        out = os.path.join(OUTPUT_TARGET, base)
    merged.to_csv(out, index=False)
    print(f"Augmented table written to: {out}")

if __name__ == '__main__':
    main()
