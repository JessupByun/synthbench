#!/usr/bin/env python3
"""
Build one big CSV of all n=32 Llama MIA runs—original + 4 ablations—for your 5 ablation datasets,
and attach each split’s Column‐Shape similarity from Alfred.

Outputs:
  Model, Dataset, TrainingSize, Seed, Attacker, ROC_AUC, Average_Shape, Corr_Similarity
"""

import os
import re
import pandas as pd
import numpy as np

# ==== USER CONFIGURATION ====  
LLAMA_MIA_ROOT    = "LTM_evaluation/LTM_Gen_MIA/llama"
LLAMA_ALFRED_ROOT = "LTM_evaluation/LTM_alfred_evaluation/llama"

DATASETS = [
    "concrete-compressive-strength",
    "naval-propulsion-plant",
    "solar-flare",
    "white-wine",
    "concrete-compressive-strength_ablation_batchsize",
    "naval-propulsion-plant_ablation_batchsize",
    "solar-flare_ablation_batchsize",
    "white-wine_ablation_batchsize",
    "concrete-compressive-strength_ablation_summarystats",
    "naval-propulsion-plant_ablation_summarystats",
    "solar-flare_ablation_summarystats",
    "white-wine_ablation_summarystats",
    "concrete-compressive-strength_ablation_temp0.1",
    "naval-propulsion-plant_ablation_temp0.1",
    "solar-flare_ablation_temp0.1",
    "white-wine_ablation_temp0.1",
    "concrete-compressive-strength_ablation_temp0.5",
    "naval-propulsion-plant_ablation_temp0.5",
    "solar-flare_ablation_temp0.5",
    "white-wine_ablation_temp0.5",
]

OUTPUT_CSV = "LTM_evaluation/llama_ablation_big_table.csv"
# ============================

def parse_split_name(folder_name):
    m = re.search(r"--train--(\d+)-seed(\d+)$", folder_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def load_columnshape_metrics(ds: str, split: str):
    """
    Return (average_shape, corr_similarity) from ColumnShape_holdout.csv
    """
    path = os.path.join(LLAMA_ALFRED_ROOT, ds, split, "ColumnShape_holdout.csv")
    if not os.path.isfile(path):
        return np.nan, np.nan

    try:
        df = pd.read_csv(path, header=None, skip_blank_lines=True)
        avg_shape_row = df[df.iloc[:, 1].astype(str).str.strip() == "Average Shape"]
        corr_sim_row  = df[
            (df.iloc[:, 0].astype(str).str.strip() == "All Pairs") &
            (df.iloc[:, 1].astype(str).str.strip() == "Average Corr Similarity")
        ]

        avg_shape = float(avg_shape_row.iloc[0, 2]) if not avg_shape_row.empty else np.nan
        corr_sim  = float(corr_sim_row.iloc[0, 2])  if not corr_sim_row.empty else np.nan

        return avg_shape, corr_sim
    except Exception:
        return np.nan, np.nan

def main():
    shape_cache = {}
    rows = []

    for ds in DATASETS:
        mia_dir    = os.path.join(LLAMA_MIA_ROOT, ds)
        alfred_dir = os.path.join(LLAMA_ALFRED_ROOT, ds)

        if not os.path.isdir(mia_dir):
            print(f"[WARN] missing MIA folder:    {mia_dir}")
            continue
        if not os.path.isdir(alfred_dir):
            print(f"[WARN] missing Alfred folder: {alfred_dir}")

        for split in sorted(os.listdir(mia_dir)):
            size, seed = parse_split_name(split)
            if size != 32:
                continue

            mia_path = os.path.join(mia_dir, split, "mia_results.csv")
            if not os.path.isfile(mia_path):
                print(f"[WARN] missing mia_results.csv for {ds}/{split}")
                continue

            key = (ds, split)
            if key not in shape_cache:
                shape_cache[key] = load_columnshape_metrics(ds, split)

            avg_shape, corr_sim = shape_cache[key]

            df = pd.read_csv(mia_path)
            if "Attacker" not in df.columns:
                df = df.reset_index().rename(columns={"index": "Attacker"})

            if "ROC_AUC" in df.columns:
                roc_col = "ROC_AUC"
            else:
                cands = [c for c in df.columns if "roc" in c.lower()]
                roc_col = cands[0] if cands else None
            if roc_col is None:
                print(f"[WARN] no ROC column in {mia_path}")
                continue

            for _, r in df.iterrows():
                rows.append({
                    "Model":           "llama",
                    "Dataset":         ds,
                    "TrainingSize":    size,
                    "Seed":            seed,
                    "Attacker":        r["Attacker"],
                    "ROC_AUC":         r[roc_col],
                    "Average_Shape":   avg_shape,
                    "Corr_Similarity": corr_sim,
                })

    out_df = pd.DataFrame(rows, columns=[
        "Model", "Dataset", "TrainingSize", "Seed", "Attacker",
        "ROC_AUC", "Average_Shape", "Corr_Similarity"
    ])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {len(out_df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
