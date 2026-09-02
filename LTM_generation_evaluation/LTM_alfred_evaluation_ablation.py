#!/usr/bin/env python3
import sys, types, os, pandas as pd
# Stub out only the alpha_precision metric so it never gets imported
sys.modules['evaluator.metrics.alpha_precision'] = types.ModuleType('evaluator.metrics.alpha_precision')
sys.modules['evaluator.metrics.alpha_precision'].AlphaPrecision = type('AlphaPrecision', (), {})

from evaluator import EvaluationPipeline

# ===== USER CONFIGURATION =====
GENERATOR_NAME = "llama"

# Path to your real‐data TRAIN folder (full path, no dataset subfolder appended)
REAL_TRAIN_DIR = "LTM_data/LTM_real_data/white-wine/train"
# Path to your real‐data TEST folder (full path, no dataset subfolder appended)
REAL_TEST_DIR  = "LTM_data/LTM_real_data/white-wine/test"

# Path to the *parent* of all your synthetic splits:
# e.g. if <…>/LTM_synthetic_data/LTM_llama_synthetic_data/synth_<ds>/*_llama_default_0.csv
SYNTH_ROOT    = os.path.join("LTM_data", "LTM_synthetic_data", f"LTM_{GENERATOR_NAME}_synthetic_data")

# Where to drop Alfred outputs:
OUTPUT_ROOT   = os.path.join("LTM_evaluation", "LTM_alfred_evaluation", GENERATOR_NAME)

# List your “ablation” folder names here (matching the synth_<name> folders)
DATASET_NAMES = [
    "white-wine_ablation_batchsize",
    "white-wine_ablation_summarystats",
    "white-wine_ablation_temp0.1",
    "white-wine_ablation_temp0.5",
]
# ==============================

def infer_column_types(df: pd.DataFrame,
                       cat_threshold: int = 50,
                       rel_cardinality: float = 0.05) -> dict:
    col_types = {}
    n = len(df)
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_datetime64_any_dtype(ser):
            col_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(ser):
            uniq = ser.nunique(dropna=True)
            if uniq < cat_threshold or (uniq / n) < rel_cardinality:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numerical"
        else:
            col_types[col] = "categorical"
    return col_types

def process_dataset_alfred(dataset_name: str, generator_name: str):
    real_folder  = REAL_TRAIN_DIR
    synth_folder = os.path.join(SYNTH_ROOT, f"synth_{dataset_name}")
    output_root  = os.path.join(OUTPUT_ROOT, dataset_name)

    os.makedirs(output_root, exist_ok=True)

    real_files = sorted(f for f in os.listdir(real_folder) if f.endswith(".csv"))
    print(f"[INFO] Found {len(real_files)} real splits in {real_folder}")

    for real_fname in real_files:
        base       = os.path.splitext(real_fname)[0]
        real_path  = os.path.join(real_folder, real_fname)
        synth_fname = f"{base}_{generator_name}_default_0.csv"
        synth_path  = os.path.join(synth_folder, synth_fname)

        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synth for {base}, skipping.")
            continue

        print(f"[INFO] Evaluating subset {base} ...")
        real_df   = pd.read_csv(real_path)
        synth_df  = pd.read_csv(synth_path)

        col_types     = infer_column_types(real_df)
        target_column = real_df.columns[-1]

        config = {
            "target_column":    target_column,
            "metadata":         col_types,
            "holdout_seed":     42,
            "holdout_size":     0.2,
        }

        subset_output = os.path.join(output_root, base)
        os.makedirs(subset_output, exist_ok=True)

        pipeline = EvaluationPipeline(
            real_data=real_df,
            synth_data=synth_df,
            column_name_to_datatype=col_types,
            config=config,
            save_path=subset_output
        )
        pipeline.run_pipeline()
        print(f"[INFO] Finished {base} → {subset_output}")

def main():
    for ds in DATASET_NAMES:
        print(f"\n=== Processing dataset: {ds} ===")
        try:
            process_dataset_alfred(ds, GENERATOR_NAME)
        except Exception as e:
            print(f"[ERROR] Failed on {ds}: {e}")

if __name__ == "__main__":
    main()
