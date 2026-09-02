#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# ==== CONFIGURATION ====
GENMIA_ROOT       = "LTM_evaluation/LTM_Gen_MIA/llama"
ALFRED_ROOT       = "LTM_evaluation/LTM_alfred_evaluation/llama"
OUTPUT_CSV        = "LTM_evaluation/llama_ablation_summary_table_with_rareAUC_and_prop.csv"

BASE_DATASETS = [
    "concrete-compressive-strength",
    "naval-propulsion-plant",
    "solar-flare",
    "white-wine",
]

# Ablation labels and their folder suffixes under GENMIA_ROOT / ALFRED_ROOT
ABLATIONS = [
    ("Default model",               ""),
    ("Batch size",                  "_ablation_batchsize"),
    ("Summary stats not in prompt", "_ablation_summarystats"),
    ("Temperature = 0.1",           "_ablation_temp0.1"),
    ("Temperature = 0.5",           "_ablation_temp0.5"),
]


def load_max_rare_auc(folder_path: str):
    path = os.path.join(folder_path, "mia_results.csv")
    if not os.path.isfile(path):
        return np.nan
    df = pd.read_csv(path)
    if "Rare_Class_ROC_AUC" not in df.columns:
        return np.nan
    vals = pd.to_numeric(df["Rare_Class_ROC_AUC"], errors="coerce").dropna()
    return vals.max() if not vals.empty else np.nan


def load_proportion_closer(folder_path: str):
    path = os.path.join(folder_path, "DCR.csv")
    if not os.path.isfile(path):
        return np.nan
    try:
        df = pd.read_csv(path)
    except:
        return np.nan
    # find the column containing 'Proportion Closer'
    col = next((c for c in df.columns if 'Proportion Closer' in c), None)
    if not col:
        return np.nan
    val = pd.to_numeric(df[col], errors='coerce')
    return val.iloc[0] if not val.empty else np.nan


def collect_summary():
    records = []
    # regex for split folders: ends with --train--32-seed<seed>
    for label, suffix in ABLATIONS:
        rare_vals = []
        prop_vals = []
        for ds in BASE_DATASETS:
            base_folder = ds + suffix
            root_path = os.path.join(GENMIA_ROOT, base_folder)
            alt_path  = os.path.join(ALFRED_ROOT, base_folder)
            if not os.path.isdir(root_path):
                continue
            for split in os.listdir(root_path):
                if not split.endswith("--train--32-seed0") and "--train--32-seed" not in split:
                    continue
                mia_folder   = os.path.join(root_path, split)
                prop_folder  = os.path.join(alt_path, split)
                # rare AUC
                ra = load_max_rare_auc(mia_folder)
                if not np.isnan(ra):
                    rare_vals.append(ra)
                # proportion closer
                pr = load_proportion_closer(prop_folder)
                if not np.isnan(pr):
                    prop_vals.append(pr)
        records.append({
            'Ablation': label,
            'Mean_Rare_Class_ROC_AUC':   float(np.nanmean(rare_vals)) if rare_vals else np.nan,
            'Mean_Proportion_Closer':    float(np.nanmean(prop_vals)) if prop_vals else np.nan,
        })
    return pd.DataFrame(records)


def main():
    summary_df = collect_summary()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote summary to {OUTPUT_CSV}")
    print(summary_df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()