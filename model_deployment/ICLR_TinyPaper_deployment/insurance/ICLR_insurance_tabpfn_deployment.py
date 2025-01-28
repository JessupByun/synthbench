import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

# === Configuration: Replace with your actual dataset ===
INPUT_FILE = "your_real_data.csv"  # Replace with your dataset file path
OUTPUT_FILE = "synthetic_data.csv" # Replace with desired output file name
TARGET_COLUMN = "target"            # Replace with the target column name

# === Load real dataset ===
df = pd.read_csv(INPUT_FILE)

# Split features and target
X = df.drop(columns=[TARGET_COLUMN])  # Replace with actual feature columns
y = df[TARGET_COLUMN]                  # Replace with target column name

# Split the data (adjust test size if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize and train TabPFN model ===
model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)  # Use 'cuda' if GPU is available
model.fit(X_train, y_train)

# === Generate synthetic data ===
# Create synthetic feature values by sampling within the range of real data
synthetic_X = np.random.uniform(X.min(), X.max(), size=(100, X.shape[1]))

# Predict labels for synthetic feature values
synthetic_y = model.predict(synthetic_X)

# Combine synthetic features and labels
synthetic_df = pd.DataFrame(synthetic_X, columns=X.columns)
synthetic_df[TARGET_COLUMN] = synthetic_y

# === Save synthetic dataset ===
synthetic_df.to_csv(OUTPUT_FILE, index=False)
print(f"Synthetic data saved to {OUTPUT_FILE}")
