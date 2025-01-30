import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import KBinsDiscretizer

# === Configuration ===
INPUT_FILE = "data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_train.csv"
OUTPUT_FILE = "data/synthetic_data/private_combined_df_cty_week2/iclr_tinypaper_tabpfn_combined_df.csv"
N_BINS = 10
N_TRAIN_SAMPLES = 200
N_SYNTH_SAMPLES = 200

# Set the target column explicitly
target_col = "mask_user_pct"

# === Load real dataset ===
df = pd.read_csv(INPUT_FILE)

# Sample exactly 200 rows for training
df = df.sample(n=N_TRAIN_SAMPLES, random_state=42)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Make sure our target column is in numerical_cols (if it's truly numeric)
# If it's not in numerical_cols, and you know it's numeric, add it:
if target_col not in numerical_cols:
    numerical_cols.append(target_col)

# === Convert all numerical columns into quantile bins ===
kbins_dict = {}
binned_df = df.copy()

for col in numerical_cols:
    kbins = KBinsDiscretizer(
        n_bins=N_BINS,
        encode='ordinal',
        strategy="quantile",
        subsample=None
    )
    # Fit on the original column, then transform -> integer bins
    binned_vals = kbins.fit_transform(df[[col]]).astype(int).flatten()
    binned_df[col] = binned_vals
    kbins_dict[col] = kbins

# Convert categorical columns to category codes
for col in categorical_cols:
    binned_df[col] = binned_df[col].astype('category').cat.codes

# === Split features (X) and target (y) ===
# Remove the target column from X; keep only numeric & categorical features
X_train = binned_df.drop(columns=[target_col])
y_train = binned_df[target_col]

assert X_train.shape[0] == N_TRAIN_SAMPLES, f"Train set has {X_train.shape[0]} rows, expected {N_TRAIN_SAMPLES}."

# === Train TabPFN classifier ===
model = TabPFNClassifier(device='cpu')
model.fit(X_train, y_train)

# === Generate synthetic data (in bin space) ===
synthetic_X = np.random.randint(
    X_train.min().values,
    X_train.max().values + 1,
    size=(N_SYNTH_SAMPLES, X_train.shape[1])
)

# === Predict target bin index ===
synthetic_y_binned = model.predict(synthetic_X)

# Build synthetic DataFrame in bin form
synthetic_df = pd.DataFrame(synthetic_X, columns=X_train.columns)

############################
# Safe bin sampling function
############################
def sample_in_bin(bin_idx, edges):
    """
    Sample a float from [low, high].
    Return np.nan if the range is invalid.
    """
    n_edges = len(edges) - 1
    if 0 <= bin_idx < n_edges:
        low, high = edges[bin_idx], edges[bin_idx + 1]
        # Check for invalid or extreme ranges
        if (
            np.isnan(low) or np.isnan(high) or
            np.isinf(low) or np.isinf(high) or
            high <= low
        ):
            return np.nan
        return np.random.uniform(low, high)
    else:
        return np.nan

# Convert each numeric feature column (except target) from bins to real values
for col in numerical_cols:
    if col == target_col:
        continue  # Skip the target here
    edges = kbins_dict[col].bin_edges_[0]
    synthetic_df[col] = synthetic_df[col].apply(lambda x: sample_in_bin(x, edges))

###############################
# Convert the target from bins
###############################
target_edges = kbins_dict[target_col].bin_edges_[0]

def sample_target_in_bin(bin_idx):
    n_edges = len(target_edges) - 1
    if 0 <= bin_idx < n_edges:
        low, high = target_edges[bin_idx], target_edges[bin_idx + 1]
        if (
            np.isnan(low) or np.isnan(high) or
            np.isinf(low) or np.isinf(high) or
            high <= low
        ):
            return np.nan
        return np.random.uniform(low, high)
    return np.nan

synthetic_df[target_col] = [sample_target_in_bin(b) for b in synthetic_y_binned]

# === Restore categorical columns from integer codes ===
for col in categorical_cols:
    original_categories = df[col].astype('category').cat.categories

    def map_cat_code(x):
        return original_categories[x] if 0 <= x < len(original_categories) else np.nan

    synthetic_df[col] = synthetic_df[col].apply(map_cat_code)

# === Save synthetic dataset ===
synthetic_df.to_csv(OUTPUT_FILE, index=False)
print(f"Synthetic data saved to {OUTPUT_FILE}")
print(synthetic_df.dtypes)
print(df.dtypes)
