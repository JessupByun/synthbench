#!/usr/bin/env python3
import os
import pandas as pd

# ==== USER CONFIGURATION ====
DATASETS = {
    "MIMIC-III (Admissions)":                   "LTM_data/LTM_real_data/ADMISSIONS/test/ADMISSIONS--test--999.csv",
    "COVID-19 County Statistics": "LTM_data/LTM_real_data/county/test/county--test--999.csv",
    "NationScape Mask Usage":     "LTM_data/LTM_real_data/private_nationscape_indv_df/test/private_nationscape_indv_df--test--999.csv",
    "Voting TS":                  "LTM_data/LTM_real_data/votingTS/test/votingTS--test--999.csv",
}
# ============================

def summarize_dataset(name, path):
    if not os.path.isfile(path):
        print(f"[ERROR] Missing file: {path}")
        return None

    df = pd.read_csv(path)
    n_rows = len(df)
    n_cols = df.shape[1]

    n_categorical = sum(
        not pd.api.types.is_numeric_dtype(df[col])
        for col in df.columns
    )

    return {
        "Dataset": name,
        "N": n_rows,
        "Features": n_cols,
        "Categorical": n_categorical
    }

def main():
    records = []
    for name, path in DATASETS.items():
        rec = summarize_dataset(name, path)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records)

    print("\n% LaTeX Table: Private Dataset Overview")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Dataset & N & \\# Features & \\# Categorical \\\\")
    print("\\midrule")
    for _, row in df.iterrows():
        print(f"{row['Dataset']} & {row['N']:,} & {row['Features']} & {row['Categorical']} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    main()
