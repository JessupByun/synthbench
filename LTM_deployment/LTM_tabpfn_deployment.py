import os
import json
import numpy as np
import pandas as pd
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

def generate_synthetic_data_tabpfn(df, batch_size):
    """
    Generates synthetic data using TabPFN's unsupervised synthetic data generation method.

    1. Identify categorical vs numeric columns.
    2. Build mappings for categorical columns and encode them as integer codes.
    3. Fit TabPFNUnsupervisedModel on the encoded batch.
    4. Generate `batch_size` synthetic samples via model.generate_synthetic_data.
    5. Decode any categorical columns back to their original values.
    """
    # Make a copy to avoid mutating the original
    Xp = df.copy()

    # Identify categorical columns and build mappings
    cat_cols = []
    cat_mappings = {}
    for col in Xp.columns:
        if not pd.api.types.is_numeric_dtype(Xp[col]):
            cat_cols.append(col)
            cat = Xp[col].astype("category")
            cat_mappings[col] = list(cat.cat.categories)
            Xp[col] = cat.cat.codes.astype(float)

    # Convert everything to float32
    X_np = Xp.values.astype(np.float32)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Initialize and fit the unsupervised model
    clf = TabPFNClassifier()
    reg = TabPFNRegressor()
    model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    model.fit(X_tensor)

    # Generate synthetic data
    synth_tensor = model.generate_synthetic_data(
        n_samples=batch_size,
        t=1.0,
        n_permutations=3
    )

    # Convert back to numpy and DataFrame
    synth_np = synth_tensor.cpu().numpy()
    synthetic_df = pd.DataFrame(synth_np, columns=df.columns)

    # Round & decode categorical columns
    for col in cat_cols:
        mapping = cat_mappings[col]
        # Round to nearest code, clamp to valid range
        codes = synthetic_df[col].round().astype(int)
        codes = codes.clip(0, len(mapping) - 1)
        # Map back
        synthetic_df[col] = codes.map(lambda i: mapping[i])

    return synthetic_df

def process_csv_file_tabpfn(input_csv, output_csv, batch_size):
    """
    Reads an input CSV, shuffles it, partitions into batches of up to batch_size rows,
    generates synthetic data for each batch, concatenates the results, and writes to output_csv.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_rows = df.shape[0]

    synthetic_batches = []
    for start in range(0, n_rows, batch_size):
        batch_df = df.iloc[start:start+batch_size]
        synthetic_batch = generate_synthetic_data_tabpfn(batch_df, batch_df.shape[0])
        if synthetic_batch is None:
            print(f"No synthetic data returned for batch starting at row {start}.")
            continue
        synthetic_batches.append(synthetic_batch)

    if not synthetic_batches:
        print("No synthetic data generated for file:", input_csv)
        return

    synthetic_df = pd.concat(synthetic_batches, ignore_index=True)
    if synthetic_df.shape[0] > n_rows:
        synthetic_df = synthetic_df.iloc[:n_rows]
    if synthetic_df.shape[0] != n_rows:
        print(f"[WARNING] Synthetic row count ({synthetic_df.shape[0]}) differs from original ({n_rows}).")

    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Synthetic data saved to {output_csv}")

def process_dataset_tabpfn(dataset_name, generator_name, batch_size):
    """
    Processes all CSV files in:
      LTM_data/LTM_real_data/{dataset_name}/train/
    Writes synthetic CSVs under:
      LTM_data/LTM_synthetic_data/LTM_tabpfn_synthetic_data/synth_{dataset_name}/

    After processing, runs the validation script on the generated data.
    """
    real_data_path = os.path.join("LTM_data", "LTM_real_data", dataset_name)
    train_folder = os.path.join(real_data_path, "train")
    if not os.path.isdir(train_folder):
        print(f"[ERROR] Train folder not found: {train_folder}")
        return

    synthetic_folder = os.path.join(
        "LTM_data", "LTM_synthetic_data", "LTM_tabpfn_synthetic_data", f"synth_{dataset_name}"
    )
    os.makedirs(synthetic_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(train_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"[WARNING] No CSV files found in {train_folder}.")
        return

    for csv_file in csv_files:
        input_csv_path = os.path.join(train_folder, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        output_csv_name = f"{base_name}_{generator_name}_default_0.csv"
        output_csv_path = os.path.join(synthetic_folder, output_csv_name)
        print(f"[INFO] Processing: {input_csv_path} -> {output_csv_path}")
        process_csv_file_tabpfn(input_csv_path, output_csv_path, batch_size=batch_size)

    # --- VALIDATION STEP ---
    try:
        from validate_synthetic_data import validate_synthetic_data, logger
    except ImportError as e:
        print(f"Error importing validation function: {e}")
        return

    real_data_dir = os.path.join(real_data_path, "train")
    synthetic_data_dir = synthetic_folder
    output_file = os.path.join(synthetic_folder, f"{dataset_name}_validation_results.json")

    validation_results = validate_synthetic_data(real_data_dir, synthetic_data_dir)

    passed = sum(1 for r in validation_results if r["validation_passed"])
    total = len(validation_results)
    logger.info(f"Validation complete: {passed}/{total} synthetic datasets passed all checks")

    for result in validation_results:
        if not result["validation_passed"]:
            logger.warning(f"Issues with {result['synthetic_file']}:")
            for issue in result["issues"]:
                logger.warning(f"  - {issue}")

    try:
        with open(output_file, "w") as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")

if __name__ == "__main__":
    dataset_name = "airfoil-self-noise"
    process_dataset_tabpfn(dataset_name, generator_name="tabpfn", batch_size=200)
