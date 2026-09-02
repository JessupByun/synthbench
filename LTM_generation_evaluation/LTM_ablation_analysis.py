#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np

# ==== USER CONFIGURATION ====
BIGTABLE_CSV       = "LTM_evaluation/llama_ablation_big_table.csv"
OUTPUT_CSV         = "LTM_evaluation/llama_ablation_summary_table.csv"
LLAMA_MIA_ROOT     = "LTM_evaluation/LTM_Gen_MIA/llama"
LLAMA_ALFRED_ROOT  = "LTM_evaluation/LTM_alfred_evaluation/llama"

BASE_DATASETS = [
    "concrete-compressive-strength",
    "naval-propulsion-plant",
    "solar-flare",
    "white-wine",
]

ABLATIONS = [
    ("Default model",                  ""),
    ("Batch size",        "_ablation_batchsize"),
    ("Summary stats not in prompt", "_ablation_summarystats"),
    ("Temperature = 0.1",   "_ablation_temp0.1"),
    ("Temperature = 0.5",   "_ablation_temp0.5"),
]
# =============================

def parse_split_name(folder_name):
    """Extract (size, seed) from '...--train--<size>-seed<seed>'"""
    m = re.search(r"--train--(\d+)-seed(\d+)$", folder_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def load_shape_and_corr(ds, split):
    """Read both Average Shape and Average Corr Similarity from Alfred."""
    p = os.path.join(LLAMA_ALFRED_ROOT, ds, split, "ColumnShape_holdout.csv")
    if not os.path.isfile(p):
        return np.nan, np.nan
    df = pd.read_csv(p, header=None, skip_blank_lines=True)
    # Shape = row where second column == "Average Shape"
    r1 = df[df.iloc[:,1].astype(str).str.strip()=="Average Shape"]
    shape = float(r1.iloc[0,2]) if not r1.empty else np.nan
    # Corr = row where first column == "All Pairs"
    r2 = df[df.iloc[:,0].astype(str).str.strip()=="All Pairs"]
    corr  = float(r2.iloc[0,2]) if not r2.empty else np.nan
    return shape, corr

def aggregate_default():
    """Scan the raw MIA folders to recover the original (no‐suffix) runs."""
    best_aucs   = []
    best_shapes = []
    best_corrs  = []

    for ds in BASE_DATASETS:
        mia_dir = os.path.join(LLAMA_MIA_ROOT, ds)
        if not os.path.isdir(mia_dir):
            print(f"[WARN] missing default MIA folder: {mia_dir}")
            continue

        for split in sorted(os.listdir(mia_dir)):
            size, seed = parse_split_name(split)
            if size != 32:
                continue

            path = os.path.join(mia_dir, split, "mia_results.csv")
            if not os.path.isfile(path):
                print(f"[WARN] missing mia_results for {ds}/{split}")
                continue
            df = pd.read_csv(path)
            # rename index→Attacker if needed
            if "Attacker" not in df.columns:
                df = df.reset_index().rename(columns={"index":"Attacker"})
            # pick ROC_AUC col
            roc_col = "ROC_AUC" if "ROC_AUC" in df.columns else \
                      next((c for c in df.columns if "roc" in c.lower()), None)
            if roc_col is None:
                continue

            # best attack
            best = df.loc[df[roc_col].idxmax()]
            best_aucs.append(best[roc_col])

            # shape & corr
            shape, corr = load_shape_and_corr(ds, split)
            best_shapes.append(shape)
            best_corrs.append(corr)

    return {
        "Ablation":             "Default model",
        "Mean_Max_Attack_AUC":  float(np.mean(best_aucs)),
        "Mean_Average_Shape":   float(np.nanmean(best_shapes)),
        "Mean_Corr_Similarity": float(np.nanmean(best_corrs)),
    }

def aggregate_ablation(label, suffix, bigdf):
    keys = [ds+suffix for ds in BASE_DATASETS]
    sub  = bigdf[bigdf.Dataset.isin(keys) & (bigdf.TrainingSize==32)]
    print(f"[DEBUG] {label}: matching {keys} → {len(sub)} rows")
    if sub.empty:
        return None

    rec = {"Ablation":label}
    # group by split
    groups = sub.groupby(["Dataset","Seed"], as_index=False)
    aucs, shapes, corrs = [], [], []

    for (ds_full, seed), grp in groups:
        # best attacker
        best = grp.loc[grp.ROC_AUC.idxmax()]
        aucs.append(best.ROC_AUC)
        shapes.append(best.Average_Shape)
        if not pd.isna(best.Corr_Similarity):
            corrs.append(best.Corr_Similarity)

    rec["Mean_Max_Attack_AUC"]  = float(np.mean(aucs))
    rec["Mean_Average_Shape"]   = float(np.mean(shapes))
    rec["Mean_Corr_Similarity"] = float(np.nanmean(corrs)) if corrs else np.nan
    return rec

def main():
    # load your big table (which has only the 4×4 ablations)
    bigdf = pd.read_csv(BIGTABLE_CSV)

    records = []
    # first the default row
    records.append(aggregate_default())

    # then each ablation
    for label, suffix in ABLATIONS[1:]:
        rec = aggregate_ablation(label, suffix, bigdf)
        if rec:
            records.append(rec)

    summary = pd.DataFrame(records, columns=[
        "Ablation",
        "Mean_Max_Attack_AUC",
        "Mean_Average_Shape",
        "Mean_Corr_Similarity",
    ])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Wrote summary table to {OUTPUT_CSV}")
    print(summary.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    main()
