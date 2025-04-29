import os
import pandas as pd
import numpy as np
from synth_mia.attackers import (
    gen_lra, dcr, dpi, logan, dcr_diff, domias,
    mc, density_estimate, local_neighborhood, classifier
)
from sklearn.metrics import roc_auc_score

# Determine project root (assuming script is in <project>/LTM_generation_evaluation)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

def get_attackers():
    """Instantiate membership-inference attackers with default hyperparameters."""
    return [
        gen_lra(hyper_parameters={"closest_compare_n": 1}),
        dcr(),
        dpi(),
        logan(),
        dcr_diff(),
        domias(),
        mc(),
        density_estimate(hyper_parameters={"method": "kde"}),
        local_neighborhood(),
        classifier(),
    ]

def load_numeric_array(path: str) -> np.ndarray:
    """
    Load a CSV into a numpy array containing only numeric columns.
    Drops non-numeric columns so that KDE-based attackers won’t error.
    """
    df = pd.read_csv(path)
    numeric_df = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(numeric_df.columns)
    if dropped:
        print(f"[WARN] Dropped non-numeric columns for MIA: {dropped}")
    return numeric_df.values

def process_dataset_genmia(dataset_name: str, generator_name: str):
    """
    Automate GenMIA evaluations for each train split in a dataset.

    - mem: each CSV under LTM_data/LTM_real_data/{dataset_name}/train/
    - non_mem & ref: the single CSV under LTM_data/LTM_real_data/{dataset_name}/test/
    - synth: matching CSVs under
        LTM_data/LTM_synthetic_data/LTM_{generator_name}_synthetic_data/synth_{dataset_name}/

    Results go to:
      LTM_evaluation/LTM_GenMIA/{dataset_name}/{generator_name}/{base}/mia_results.csv
    """
    # Absolute paths to your data
    train_folder = os.path.join(PROJECT_ROOT, "LTM_data", "LTM_real_data", dataset_name, "train")
    test_folder  = os.path.join(PROJECT_ROOT, "LTM_data", "LTM_real_data", dataset_name, "test")

    if not os.path.isdir(train_folder):
        raise NotADirectoryError(f"Train folder not found: {train_folder}")
    if not os.path.isdir(test_folder):
        raise NotADirectoryError(f"Test folder not found: {test_folder}")

    # Load test CSV as both non_mem and ref
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
    if len(test_files) != 1:
        raise ValueError(f"Expected one test CSV in {test_folder}, found {len(test_files)}")
    test_path = os.path.join(test_folder, test_files[0])
    non_mem = load_numeric_array(test_path)
    ref     = non_mem.copy()

    # Synthetic data folder
    synth_folder = os.path.join(
        PROJECT_ROOT,
        "LTM_data",
        "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    if not os.path.isdir(synth_folder):
        raise NotADirectoryError(f"Synthetic folder not found: {synth_folder}")

    # Output root for GenMIA results
    output_root = os.path.join(
        PROJECT_ROOT,
        "LTM_evaluation", "LTM_Gen_MIA",
        generator_name, dataset_name
    )
    os.makedirs(output_root, exist_ok=True)

    attackers = get_attackers()

    # Loop through each train CSV (mem)
    for mem_fname in sorted(os.listdir(train_folder)):
        if not mem_fname.endswith('.csv'):
            continue
        base = os.path.splitext(mem_fname)[0]

        # Member data
        mem = load_numeric_array(os.path.join(train_folder, mem_fname))

        # Corresponding synthetic CSV
        synth_fname = f"{base}_{generator_name}_default_0.csv"
        synth_path  = os.path.join(synth_folder, synth_fname)
        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synthetic for {base}: {synth_path}")
            continue
        synth = load_numeric_array(synth_path)

        # Run attacks & compute ROC AUC manually
        results = {}
        for attacker in attackers:
            scores, labels = attacker.attack(mem, non_mem, synth, ref)
            
            # Evaluate the attack using the ROC metric.
            eval_results = attacker.eval(scores, labels, metrics=['roc'])
    
            # Save the evaluation results with the attacker name as key.
            results[attacker.name] = eval_results

        # Save results
        df = pd.DataFrame.from_dict(results, orient='index', columns=['roc_auc'])
        save_dir = os.path.join(output_root, base)
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, 'mia_results.csv')
        df.to_csv(out_csv)
        print(f"[INFO] Saved MIA results to {out_csv}")

def main():
    dataset_name   = "abalone"   # e.g. "abalone"
    generator_name = "llama"     # e.g. "tabpfn" or "llama"
    process_dataset_genmia(dataset_name, generator_name)

if __name__ == "__main__":
    main()
