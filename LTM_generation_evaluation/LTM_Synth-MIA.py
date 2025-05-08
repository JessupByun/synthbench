import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from synth_mia.attackers import (
    gen_lra, dcr, dpi, logan, dcr_diff, domias,
    mc, density_estimate, local_neighborhood, classifier
)

# Determine project root (assumes this script is in <project>/LTM_generation_evaluation/)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


def get_attackers():
    """Instantiate membership-inference attackers with default hyperparameters."""
    return [
        gen_lra(hyper_parameters={"closest_compare_n": 1}),
        dcr(), dpi(), logan(), dcr_diff(), domias(),
        mc(), density_estimate(hyper_parameters={"method": "kde"}),
        local_neighborhood(), classifier(),
    ]


def process_dataset_genmia(dataset_name: str, generator_name: str):
    """
    For each CSV in LTM_data/LTM_real_data/{dataset_name}/train/:
      - split 50/50 into mem vs ref (seed=42)
      - load non_mem as the single CSV under LTM_data/LTM_real_data/{dataset_name}/test/
      - load matching synthetic CSV
      - encode categoricals + keep numerics
      - run membership-inference attacks
      - compute ROC AUC
      - save results
    """
    # Paths
    train_folder = os.path.join(PROJECT_ROOT, "LTM_data", "LTM_real_data", dataset_name, "train")
    test_folder  = os.path.join(PROJECT_ROOT, "LTM_data", "LTM_real_data", dataset_name, "test")
    synth_folder = os.path.join(
        PROJECT_ROOT,
        "LTM_data", "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )
    output_root = os.path.join(PROJECT_ROOT, "LTM_evaluation", "LTM_Gen_MIA", generator_name, dataset_name)
    os.makedirs(output_root, exist_ok=True)

    # Check folders
    if not os.path.isdir(train_folder):
        raise NotADirectoryError(f"Train folder not found: {train_folder}")
    if not os.path.isdir(test_folder):
        raise NotADirectoryError(f"Test folder not found: {test_folder}")
    if not os.path.isdir(synth_folder):
        raise NotADirectoryError(f"Synthetic folder not found: {synth_folder}")

    # Identify the one test CSV
    test_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.csv')]
    if len(test_files) != 1:
        raise ValueError(f"Expected exactly one test CSV in {test_folder}, found {len(test_files)}")
    non_mem_path = os.path.join(test_folder, test_files[0])

    attackers = get_attackers()

    # Loop over each train CSV
    for train_fname in sorted(os.listdir(train_folder)):
        if not train_fname.lower().endswith('.csv'):
            continue

        base = os.path.splitext(train_fname)[0]
        real_path  = os.path.join(train_folder, train_fname)
        synth_fname = f"{base}_{generator_name}_default_0.csv"
        synth_path  = os.path.join(synth_folder, synth_fname)
        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synthetic for {base}: {synth_path}")
            continue

        # 1) Load and split real data
        df = pd.read_csv(real_path)
        mem_df, ref_df = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)

        # 2) Identify numeric and categorical columns from real data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        # 3) Fit label encoders on real data for categorical columns
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            le.fit(df[c].astype(str))
            encoders[c] = le
            # transform mem and ref
            mem_df[c] = mem_df[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)
            ref_df[c] = ref_df[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

        # 4) Load non_mem fresh for each iteration and encode it
        non_mem_df = pd.read_csv(non_mem_path)
        for c in cat_cols:
            le = encoders[c]
            non_mem_df[c] = non_mem_df[c].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )

        # 5) Load synthetic and encode categoricals
        synth_df = pd.read_csv(synth_path)
        for c in cat_cols:
            le = encoders[c]
            if c in synth_df.columns:
                synth_df[c] = synth_df[c].astype(str).map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
            else:
                synth_df[c] = -1

        # 6) Build numpy arrays for attacks
        feature_cols = num_cols + cat_cols
        mem      = mem_df[feature_cols].values
        ref      = ref_df[feature_cols].values
        non_mem  = non_mem_df[feature_cols].values
        synth    = synth_df[feature_cols].values

        # DEBUG: check shapes before attacking
        print(f"[DEBUG] {base} shapes – mem: {mem.shape}, non_mem: {non_mem.shape}, synth: {synth.shape}, ref: {ref.shape}")

        # 7) Run attacks and compute ROC AUC
        results = {}
        for attacker in attackers:
            try:
                scores, raw_labels = attacker.attack(mem, non_mem, synth, ref)
                true_labels = (raw_labels > 0.5).astype(int)
                auc = roc_auc_score(true_labels, scores)
            except Exception as e:
                print(f"[ERROR] {attacker.name} on {base}: {e}")
                auc = float('nan')
            results[attacker.name] = auc

        # 8) Save results
        df_out = pd.DataFrame.from_dict(results, orient='index', columns=['roc_auc'])
        save_dir = os.path.join(output_root, base)
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, 'mia_results.csv')
        df_out.to_csv(out_csv)
        print(f"[INFO] Saved MIA results to {out_csv}")


def main():
    dataset_name   = "abalone"   # e.g. "abalone"
    generator_name = "llama"     # e.g. "tabpfn" or "llama"
    process_dataset_genmia(dataset_name, generator_name)


if __name__ == "__main__":
    main()
