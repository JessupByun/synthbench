import os
import json
import numpy as np
import pandas as pd
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

# Resolve project root (assumes this file lives in <project>/LTM_generation_evaluation/)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

def generate_synthetic_data_tabpfn(df, batch_size):
    """
    Generates synthetic data using TabPFN's unsupervised method.
    """
    Xp = df.copy()
    cat_cols = []
    cat_mappings = {}

    # encode categoricals
    for col in Xp.columns:
        if not pd.api.types.is_numeric_dtype(Xp[col]):
            cat_cols.append(col)
            cat = Xp[col].astype("category")
            cat_mappings[col] = list(cat.cat.categories)
            Xp[col] = cat.cat.codes.astype(float)

    X_np     = Xp.values.astype(np.float32)
    X_tensor = torch.from_numpy(X_np)

    clf   = TabPFNClassifier()
    reg   = TabPFNRegressor()
    model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    model.fit(X_tensor)

    synth_tensor = model.generate_synthetic_data(
        n_samples=batch_size,
        t=1.0,
        n_permutations=3
    )

    synth_np     = synth_tensor.cpu().numpy()
    synthetic_df = pd.DataFrame(synth_np, columns=df.columns)

    # decode categoricals
    for col in cat_cols:
        mapping = cat_mappings[col]
        codes   = synthetic_df[col].round().astype(int).clip(0, len(mapping)-1)
        synthetic_df[col] = codes.map(lambda i: mapping[i])

    return synthetic_df

def process_csv_file_tabpfn(input_csv, output_csv, batch_size):
    """
    Read input_csv, shuffle & batch, generate synthetic data, write to output_csv.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_rows = len(df)

    synthetic_batches = []
    for start in range(0, n_rows, batch_size):
        batch_df = df.iloc[start:start+batch_size]
        synthetic_batch = generate_synthetic_data_tabpfn(batch_df, batch_df.shape[0])
        if synthetic_batch is None:
            print(f"No synthetic returned for batch at row {start}")
            continue
        synthetic_batches.append(synthetic_batch)

    if not synthetic_batches:
        print("[ERROR] No synthetic data generated for", input_csv)
        return

    synthetic_df = pd.concat(synthetic_batches, ignore_index=True)
    # ensure same row count
    synthetic_df = synthetic_df.iloc[:n_rows]
    if len(synthetic_df) != n_rows:
        print(f"[WARN] Synthetic rows {len(synthetic_df)} != original {n_rows}")

    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved synthetic to {output_csv}")

def process_dataset_tabpfn(dataset_name, generator_name="tabpfn", batch_size=200):
    """
    For each CSV in LTM_data/LTM_real_data/{dataset_name}/train/,
    generate synthetic via TabPFN and write to
    LTM_data/LTM_synthetic_data/LTM_{generator_name}_synthetic_data/synth_{dataset_name}/
    Then run validate_synthetic_data on them.
    """
    # absolute paths
    real_train = os.path.join(PROJECT_ROOT, "LTM_data", "LTM_real_data", dataset_name, "train")
    if not os.path.isdir(real_train):
        print(f"[ERROR] Train folder not found: {real_train}")
        return

    synth_folder = os.path.join(
        PROJECT_ROOT,
        "LTM_data", "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    os.makedirs(synth_folder, exist_ok=True)

    for fname in sorted(os.listdir(real_train)):
        if not fname.lower().endswith(".csv"):
            continue
        in_csv  = os.path.join(real_train, fname)
        base    = os.path.splitext(fname)[0]
        out_csv = os.path.join(synth_folder, f"{base}_{generator_name}_default_0.csv")

        print(f"[INFO] Generating: {in_csv} → {out_csv}")
        process_csv_file_tabpfn(in_csv, out_csv, batch_size)

    # --- validation ---
    try:
        from validate_synthetic_data import validate_synthetic_data, logger
    except ImportError as e:
        print(f"[ERROR] Could not import validator: {e}")
        return

    validation_results = validate_synthetic_data(real_train, synth_folder)

    passed = sum(1 for r in validation_results if r["validation_passed"])
    total  = len(validation_results)
    logger.info(f"Validation: {passed}/{total} passed")

    # save JSON
    out_json = os.path.join(synth_folder, f"{dataset_name}_validation_results.json")
    try:
        with open(out_json, "w") as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Saved validation to {out_json}")
    except Exception as e:
        logger.error(f"Error writing {out_json}: {e}")

if __name__ == "__main__":
    # change to any of your datasets
    process_dataset_tabpfn("airfoil-self-noise", generator_name="tabpfn", batch_size=200)
