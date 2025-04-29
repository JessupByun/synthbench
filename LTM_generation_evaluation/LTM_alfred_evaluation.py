import os
import pandas as pd
from evaluator import EvaluationPipeline

def infer_column_types(df: pd.DataFrame,
                       cat_threshold: int = 50,
                       rel_cardinality: float = 0.05) -> dict:
    """
    Infer column_name_to_datatype mapping for Alfred:
      - numerical if pandas sees it as number and cardinality is high
      - datetime if datetime dtype
      - categorical otherwise

    Args:
      df: the real-data DataFrame
      cat_threshold: absolute unique-value threshold to treat as categorical
      rel_cardinality: relative unique-value threshold (unique/n_rows)
    """
    col_types = {}
    n = len(df)
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_datetime64_any_dtype(ser):
            col_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(ser):
            uniq = ser.nunique(dropna=True)
            # treat as categorical if low cardinality
            if uniq < cat_threshold or (uniq / n) < rel_cardinality:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numerical"
        else:
            col_types[col] = "categorical"
    return col_types

def process_dataset_alfred(dataset_name: str, generator_name: str):
    """
    Automate Alfred evaluation for all real/synthetic CSV pairs.
    Writes outputs under LTM_data/LTM_alfred_evaluation/{dataset_name}/{generator_name}/{base}/
    """
    real_folder = os.path.join("LTM_data", "LTM_real_data", dataset_name, "train")
    synth_folder = os.path.join(
        "LTM_data",
        "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    output_root = os.path.join("LTM_data", "LTM_alfred_evaluation",
                               dataset_name, generator_name)

    if not os.path.isdir(real_folder):
        print(f"[ERROR] Real data folder not found: {real_folder}")
        return
    if not os.path.isdir(synth_folder):
        print(f"[ERROR] Synthetic data folder not found: {synth_folder}")
        return

    for real_fname in sorted(os.listdir(real_folder)):
        if not real_fname.lower().endswith(".csv"):
            continue

        base = os.path.splitext(real_fname)[0]
        real_path = os.path.join(real_folder, real_fname)
        synth_fname = f"{base}_{generator_name}_default_0.csv"
        synth_path = os.path.join(synth_folder, synth_fname)

        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synthetic file for {real_fname}: {synth_path}")
            continue

        # load data
        real_df = pd.read_csv(real_path)
        synth_df = pd.read_csv(synth_path)

        # infer column types & set target
        col_types = infer_column_types(real_df)
        target_column = real_df.columns[-1]

        # minimal Alfred config
        config = {
            "target_column": target_column,
            "metadata": col_types,
        }

        # build save path
        save_path = os.path.join(output_root, base)
        os.makedirs(save_path, exist_ok=True)

        print(f"[INFO] Running Alfred on {real_fname} vs {synth_fname} → {save_path}")
        pipeline = EvaluationPipeline(
            real_data=real_df,
            synth_data=synth_df,
            column_name_to_datatype=col_types,
            config=config,
            save_path=save_path
        )
        pipeline.run_pipeline()

def main():
    # specify dataset and generator
    dataset_name = "abalone"    # e.g. "abalone"
    generator_name = "llama"    # e.g. "tabpfn" or "llama"

    process_dataset_alfred(dataset_name, generator_name)

if __name__ == "__main__":
    main()