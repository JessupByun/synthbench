import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import KBinsDiscretizer

# === Configuration ===
INPUT_FILE = "data/real_data/insurance/insurance_train.csv"  # Path to real dataset (should be train only)
OUTPUT_FILE = "data/synthetic_data/insurance/iclr_tinypaper_tabpfn_insurance.csv"  # Output synthetic data
N_BINS = 10            # Number of bins (using quantile-based binning)
N_TRAIN_SAMPLES = 200  # Desired number of rows for training
N_SYNTH_SAMPLES = 200  # Number of synthetic rows to generate

# === Load real dataset ===
df = pd.read_csv(INPUT_FILE)

# Sample exactly 200 rows for training
df = df.sample(n=N_TRAIN_SAMPLES, random_state=42)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()  
# Expected order: ['age', 'bmi', 'children', 'charges']

categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# === Convert all numerical columns into quantile bins ===
kbins_dict = {}
binned_df = df.copy()

for col in numerical_cols:
    # Create a quantile binning transformer, with 10 bins by default
    kbins = KBinsDiscretizer(
        n_bins=N_BINS, 
        encode='ordinal', 
        strategy="quantile", 
        subsample=None
    )
    # Fit on the original column, transform, and store
    binned_vals = kbins.fit_transform(df[[col]]).astype(int).flatten()
    binned_df[col] = binned_vals
    kbins_dict[col] = kbins

# Convert categorical columns to category codes
for col in categorical_cols:
    binned_df[col] = binned_df[col].astype('category').cat.codes

# === Split features and target ===
# Use the last numeric column as the target (here, 'charges')
X_train = binned_df.drop(columns=numerical_cols[-1])
y_train = binned_df[numerical_cols[-1]]

# Confirm the training set size
assert X_train.shape[0] == N_TRAIN_SAMPLES, f"Train set has {X_train.shape[0]} rows, expected {N_TRAIN_SAMPLES}."

# === Train TabPFN classifier ===
model = TabPFNClassifier(device='cpu')
model.fit(X_train, y_train)

# === Generate synthetic data (in bin space) ===
# Sample integer bin indices from the min to max in each feature column
synthetic_X = np.random.randint(
    X_train.min().values, 
    X_train.max().values + 1, 
    size=(N_SYNTH_SAMPLES, X_train.shape[1])
)

# === Predict target bin index ===
synthetic_y_binned = model.predict(synthetic_X)

# === Convert all numeric columns (including target) from bins to real values ===
# 1) Build the synthetic feature DataFrame in bin form
synthetic_df = pd.DataFrame(synthetic_X, columns=X_train.columns)

# 2) Invert each numeric feature column to a random real number in its bin range
for col in numerical_cols[:-1]:  # all numeric columns except the last one (target)
    edges = kbins_dict[col].bin_edges_[0] 
    n_edges = len(edges) - 1

    def sample_in_bin(bin_idx):
        if 0 <= bin_idx < n_edges:
            low, high = edges[bin_idx], edges[bin_idx+1]
            return np.random.uniform(low, high)
        else:
            return np.nan

    synthetic_df[col] = synthetic_df[col].apply(sample_in_bin)

# 3) Convert the target bin index to a random real value in its bin's range
target_col = numerical_cols[-1]
target_edges = kbins_dict[target_col].bin_edges_[0]
n_target_edges = len(target_edges) - 1

def sample_target_in_bin(bin_idx):
    if 0 <= bin_idx < n_target_edges:
        low, high = target_edges[bin_idx], target_edges[bin_idx+1]
        return np.random.uniform(low, high)
    else:
        return np.nan

synthetic_target_col = [sample_target_in_bin(bin_idx) for bin_idx in synthetic_y_binned]
synthetic_df[target_col] = synthetic_target_col

# === Restore categorical columns from integer codes ===
for col in categorical_cols:
    original_categories = df[col].astype('category').cat.categories

    def map_cat_code(x):
        return original_categories[x] if 0 <= x < len(original_categories) else np.nan

    synthetic_df[col] = synthetic_df[col].apply(map_cat_code)

# === POST-PROCESSING: Convert numeric columns to desired format ===
# numeric_cols = ['age', 'bmi', 'children', 'charges']

# age -> int
synthetic_df["age"] = synthetic_df["age"].round().astype(int)

# bmi -> float, 2 decimals
synthetic_df["bmi"] = synthetic_df["bmi"].round(2)

# children -> int
synthetic_df["children"] = synthetic_df["children"].round().astype(int)

# charges -> float, 2 decimals
synthetic_df["charges"] = synthetic_df["charges"].round(2)

# === Save synthetic dataset ===
synthetic_df.to_csv(OUTPUT_FILE, index=False)
print(f"Synthetic data saved to {OUTPUT_FILE}")