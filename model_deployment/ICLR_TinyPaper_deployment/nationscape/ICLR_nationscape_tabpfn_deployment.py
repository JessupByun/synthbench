import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import KBinsDiscretizer

# === Configuration ===
INPUT_FILE = "data/real_data/private_nationscape_indv/private_nationscape_indv_train.csv"
OUTPUT_FILE = "data/synthetic_data/private_nationscape_indv/iclr_tinypaper_tabpfn_nationscape.csv"
N_BINS = 10
N_TRAIN_SAMPLES = 200
N_SYNTH_SAMPLES = 200

# Set the target column explicitly
target_col = "worn"

# === Load real dataset ===
df = pd.read_csv(INPUT_FILE)

# Sample exactly 200 rows for training
df = df.sample(n=N_TRAIN_SAMPLES, random_state=42)

# --- Convert target column to numeric if necessary ---
if target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
    # First, convert to string in case values are mixed
    df[target_col] = df[target_col].astype(str)
    # Replace "True"/"False" with 1/0 (adjust if needed for your data)
    df[target_col] = df[target_col].replace({'True': 1, 'False': 0})
    # Now convert to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='raise')

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Ensure target is in numerical_cols if now numeric
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
    # Fit on the original column, then transform to integer bins
    binned_vals = kbins.fit_transform(df[[col]]).astype(int).flatten()
    binned_df[col] = binned_vals
    kbins_dict[col] = kbins

# Convert categorical columns to integer codes (for modeling)
for col in categorical_cols:
    binned_df[col] = binned_df[col].astype('category').cat.codes

# === Split features (X) and target (y) ===
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

####################################
# Safe bin sampling function (for numeric features)
####################################
def sample_in_bin(bin_idx, edges):
    """
    Sample a float from [low, high].
    Return np.nan if the range is invalid.
    """
    n_edges = len(edges) - 1
    if 0 <= bin_idx < n_edges:
        low, high = edges[bin_idx], edges[bin_idx + 1]
        if (
            np.isnan(low) or np.isnan(high) or
            np.isinf(low) or np.isinf(high) or
            high <= low
        ):
            return np.nan
        return np.random.uniform(low, high)
    else:
        return np.nan

# Convert each numeric feature column (except target) from bins back to real values
for col in numerical_cols:
    if col == target_col:
        continue  # Skip target here; handle separately
    edges = kbins_dict[col].bin_edges_[0]
    synthetic_df[col] = synthetic_df[col].apply(lambda x: sample_in_bin(x, edges))

####################################
# Convert the target column from bins back to real values
####################################
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

####################################
# Restore categorical columns from integer codes
####################################
for col in categorical_cols:
    # Get original category order from the original dataframe
    original_categories = df[col].astype('category').cat.categories
    # Force synthetic values to be integers by rounding/clipping.
    # (Assumes synthetic values for categorical columns are generated in the same integer range.)
    synthetic_df[col] = synthetic_df[col].round().astype(int)
    def map_cat_code(x):
        if 0 <= x < len(original_categories):
            return original_categories[x]
        else:
            return np.nan
    synthetic_df[col] = synthetic_df[col].apply(map_cat_code)

# === Save synthetic dataset ===
synthetic_df.to_csv(OUTPUT_FILE, index=False)
print(f"Synthetic data saved to {OUTPUT_FILE}")
print("Synthetic dtypes:")
print(synthetic_df.dtypes)
print("Original dtypes:")
print(df.dtypes)