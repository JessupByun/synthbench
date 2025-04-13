import os
import numpy as np
import pandas as pd
import torch

from tabpfn_extensions import unsupervised
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

def generate_synthetic_data_tabpfn(df, batch_size):
    """
    Generates synthetic data using TabPFN's unsupervised synthetic data generation method,
    following the PriorLabs unsupervised tutorial (https://priorlabs.ai/tutorials/unsupervised/).

    For the given DataFrame (with the rightmost column as the target), the function:
      1. Splits the data into features (all columns except the last) and target (the last column).
      2. For any non-numeric columns in features or the target, converts them to integer codes.
      3. Converts the processed features and target to numpy arrays of type float32.
      4. Converts these arrays into torch tensors.
      5. Initializes the TabPFN unsupervised model using a TabPFNClassifier for both classifier and regressor.
         Here, n_estimators specifies the number of base models in the ensemble. The official example uses 3,
         which is a good starting point for low-data scenarios. Depending on the complexity of your dataset,
         you might experiment with a different number, but 3 is generally sufficient.
      6. Sets attribute_names to the feature column names and feature_indices to all their indices.
      7. Runs the synthetic experiment with n_samples = 3 × (number of rows in the batch) and then slices to 
         the original batch size.
      8. Returns a synthetic DataFrame with the same column order as the input.

    Returns:
        Synthetic DataFrame, or None if synthetic data could not be generated.
    """
    # Define target and feature columns.
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    # Convert non-numeric features to integer codes.
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes

    # Convert non-numeric target to integer codes, if needed.
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = df[target_col].astype("category").cat.codes

    # Now convert data to numpy arrays as float32.
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Convert to torch tensors.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Initialize TabPFN models.
    # n_estimators determines how many ensemble members TabPFN uses.
    # The official example uses 3; this is typically a good default in low-data regimes.
    clf = TabPFNClassifier(n_estimators=3)
    reg = TabPFNRegressor(n_estimators=3)  
    model_unsupervised = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

    # Set feature names and indices.
    attribute_names = list(feature_cols)
    feature_indices = list(range(len(attribute_names)))

    # Create and run the synthetic experiment.
    exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(task_type="unsupervised")
    results = exp_synthetic.run(
        tabpfn=model_unsupervised,
        X=X_tensor,
        y=y_tensor,
        attribute_names=attribute_names,
        temp=1.0,
        n_samples=batch_size,
        indices=feature_indices,
    )

    # Check if synthetic data was returned.
    if results is None or "synthetic_data" not in results:
        print("No synthetic data returned from the unsupervised experiment.")
        return None

    # Extract synthetic data.
    synthetic_tensor = results["synthetic_data"]
    synthetic_np = synthetic_tensor.cpu().numpy() if torch.is_tensor(synthetic_tensor) else np.array(synthetic_tensor)

    # Slice synthetic data to the original number of rows (in case of oversampling).
    if synthetic_np.shape[0] > X_tensor.shape[0]:
        synthetic_np = synthetic_np[:X_tensor.shape[0], :]

    synthetic_df = pd.DataFrame(synthetic_np, columns=list(feature_cols) + [target_col])
    return synthetic_df

def process_csv_file_tabpfn(input_csv, output_csv, batch_size):
    """
    Reads an input CSV, shuffles it using seed=42, and partitions it into batches of up to batch_size rows.
    For each batch, it calls generate_synthetic_data_tabpfn to generate synthetic data, then concatenates the results
    and writes the final synthetic DataFrame to output_csv.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_rows = df.shape[0]
    
    synthetic_batches = []
    for start in range(0, n_rows, batch_size):
        batch_df = df.iloc[start:start+batch_size]
        synthetic_batch = generate_synthetic_data_tabpfn(batch_df, batch_size)
        if synthetic_batch is None:
            print(f"No synthetic data returned for batch starting at row {start}.")
            continue
        synthetic_batches.append(synthetic_batch)
        
    if len(synthetic_batches) == 0:
        print("No synthetic data generated for file:", input_csv)
        return
    
    synthetic_df = pd.concat(synthetic_batches, ignore_index=True)
    if synthetic_df.shape[0] > n_rows:
        synthetic_df = synthetic_df.iloc[:n_rows, :]
    if synthetic_df.shape[0] != n_rows:
        print(f"[WARNING] Synthetic row count ({synthetic_df.shape[0]}) differs from original ({n_rows}).")
    
    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Synthetic data saved to {output_csv}")

def process_dataset_tabpfn(dataset_name, generator_name, batch_size):
    """
    Processes all CSV files in:
      LTM_data/LTM_real_data/{dataset_name}/train/
    Synthetic CSV outputs are saved under:
      LTM_data/LTM_synthetic_data/LTM_tabpfn_synthetic_data/synth_{dataset_name}/
    Each file is named:
      {original_csv_filename}_{generator_name}_default_0.csv
    """
    real_data_path = os.path.join("LTM_data", "LTM_real_data", dataset_name)
    train_folder = os.path.join(real_data_path, "train")
    if not os.path.isdir(train_folder):
        print(f"[ERROR] Train folder not found: {train_folder}")
        return
    
    synthetic_folder = os.path.join("LTM_data", "LTM_synthetic_data", "LTM_tabpfn_synthetic_data", f"synth_{dataset_name}")
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

if __name__ == "__main__":
    dataset_name = "abalone"
    process_dataset_tabpfn(dataset_name, generator_name="tabpfn", batch_size=200)
