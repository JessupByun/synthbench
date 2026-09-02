#!/usr/bin/env python3
"""
Merge script for Gen-MIA (with/without reference) and Alfred evaluator results
into one big untidy table for public datasets.

USAGE:
  1. Update the placeholders in the `main()` function under CONFIGURATION.
  2. Run `python merge_big_data.py`.
"""
import os
import re
import pandas as pd
import numpy as np


def parse_split_name(folder_name):
    """
    Extract (training_size, seed) from a folder name of the form:
        <dataset>--train--<size>-seed<seed>
    Returns (size:int, seed:int) or (None, None) on parse failure.
    """
    match = re.search(r'--train--(\d+)-seed(\d+)', folder_name)
    if not match:
        return None, None
    size = int(match.group(1))
    seed = int(match.group(2))
    return size, seed


def collect_alfred(alfred_root):
    """
    Traverse Alfred evaluation folders and extract key metrics:
      - ColumnShape_holdout.csv → avg_corr_similarity, avg_shape

    Returns a DataFrame with columns:
    [Model, Dataset, TrainingSize, Seed, avg_corr_similarity, avg_shape]
    """
    records = []
    for dirpath, _, files in os.walk(alfred_root):
        # Case-insensitive check for ColumnShape_holdout.csv
        if not any(f.lower() == 'columnshape_holdout.csv' for f in files):
            continue

        split_folder = os.path.basename(dirpath)
        size, seed = parse_split_name(split_folder)
        if size is None:
            continue

        rel = os.path.relpath(dirpath, alfred_root).split(os.sep)
        if len(rel) < 2:
            continue
        model, dataset = rel[0], rel[1]

        rec = {
            'Model': model,
            'Dataset': dataset,
            'TrainingSize': size,
            'Seed': seed,
            'avg_corr_similarity': np.nan,
            'avg_shape': np.nan
        }

        # Locate the actual file name in a case-insensitive manner
        cs_fname = next(f for f in files if f.lower() == 'columnshape_holdout.csv')
        cs_path = os.path.join(dirpath, cs_fname)

        try:
            df = pd.read_csv(cs_path)
            # Average Corr Similarity
            mask_corr = (
                df['Column'].str.strip().eq('All Pairs') &
                df['Metric'].str.strip().eq('Average Corr Similarity')
            )
            if mask_corr.any():
                rec['avg_corr_similarity'] = df.loc[mask_corr, 'Score'].iat[0]
            # Average Shape
            mask_shape = (
                df['Column'].str.strip().eq('All') &
                df['Metric'].str.strip().eq('Average Shape')
            )
            if mask_shape.any():
                rec['avg_shape'] = df.loc[mask_shape, 'Score'].iat[0]
        except Exception:
            pass

        records.append(rec)

    return pd.DataFrame.from_records(records)


def collect_mia(root_dir, reference_flag):
    """
    Traverse a Gen-MIA result folder (with or without reference) and read
    `mia_results.csv` in each split. Emits one row per attacker:

    Columns:
      [Model, Dataset, TrainingSize, Seed, Reference,
       Attacker, AUC_Score, TPR@FPR=0, TPR@FPR=0.001,
       TPR@FPR=0.01, TPR@FPR=0.1]
    """
    records = []
    for dirpath, _, files in os.walk(root_dir):
        if 'mia_results.csv' not in files:
            continue

        split_folder = os.path.basename(dirpath)
        size, seed = parse_split_name(split_folder)
        if size is None:
            continue

        rel = os.path.relpath(dirpath, root_dir).split(os.sep)
        if len(rel) < 2:
            continue
        model, dataset = rel[0], rel[1]

        try:
            df = pd.read_csv(os.path.join(dirpath, 'mia_results.csv'))
        except Exception:
            continue

        attacker_col = df.columns[0]
        for _, row in df.iterrows():
            records.append({
                'Model': model,
                'Dataset': dataset,
                'TrainingSize': size,
                'Seed': seed,
                'Reference': reference_flag,
                'Attacker': row[attacker_col],
                'AUC_Score':    row.get('auc_roc',          np.nan),
                'TPR@FPR=0':    row.get('tpr_at_fpr_0',     np.nan),
                'TPR@FPR=0.001':row.get('tpr_at_fpr_0.001', np.nan),
                'TPR@FPR=0.01': row.get('tpr_at_fpr_0.01',  np.nan),
                'TPR@FPR=0.1':  row.get('tpr_at_fpr_0.1',   np.nan),
            })

    cols = ['Model','Dataset','TrainingSize','Seed',
            'Reference','Attacker',
            'AUC_Score','TPR@FPR=0','TPR@FPR=0.001',
            'TPR@FPR=0.01','TPR@FPR=0.1']
    return pd.DataFrame.from_records(records, columns=cols)


def main():
    # ===== USER CONFIGURATION =====
    ALFRED_DIR       = 'LTM_evaluation/LTM_alfred_evaluation'
    GENMIA_REF_DIR   = 'LTM_evaluation/LTM_Gen_MIA'
    GENMIA_NOREf_DIR = 'LTM_evaluation/LTM_Gen_MIA_no_ref'
    OUTPUT_FILE      = 'LTM_evaluation/big_data_public.csv'
    # ==============================

    print("Collecting Alfred metrics...")
    df_alfred = collect_alfred(ALFRED_DIR)

    print("Collecting Gen-MIA (with reference)...")
    df_ref   = collect_mia(GENMIA_REF_DIR, True)
    print("Collecting Gen-MIA (no reference)...")
    df_noref = collect_mia(GENMIA_NOREf_DIR, False)

    print("Merging datasets...")
    df_mia  = pd.concat([df_ref, df_noref], ignore_index=True)
    df_full = df_mia.merge(
        df_alfred,
        on=['Model','Dataset','TrainingSize','Seed'],
        how='left'
    )

    print(f"Writing output ({len(df_full)} rows) to {OUTPUT_FILE}...")
    df_full.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == '__main__':
    main()