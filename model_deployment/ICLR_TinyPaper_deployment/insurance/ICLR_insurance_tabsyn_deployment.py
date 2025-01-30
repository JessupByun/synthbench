import sys
import os

# Add the **outer** tabsyn directory to sys.path
tabsyn_path = os.path.abspath("/Users/jessupbyun/Desktop/VSCode/synthetic_data_research_winter25/tabsyn")
sys.path.append(tabsyn_path)

# Now import from the correct structure
from tabsyn import Model  # Use the proper module path

import pandas as pd

# Define input and output file names directly
INPUT_CSV = "data/real_data/insurance/insurance_train.csv"
OUTPUT_CSV = "data/synthetic_data/insurance/iclr_tinypaper_tabsyn_insurance.csv"
NUM_ROWS = 200  # Number of synthetic rows to generate

def generate_synthetic_data(input_csv, output_csv, num_rows=200):
    """Loads a CSV dataset, generates synthetic data using TabSyn, and writes to a file."""
    # Load dataset
    df = pd.read_csv(input_csv)

    # Initialize and fit TabSyn model
    model = Model()
    model.fit(df)  # Fit model on the original dataset

    # Generate synthetic data
    synthetic_data = model.sample(n=num_rows)

    # Save synthetic data to CSV
    synthetic_data.to_csv(output_csv, index=False)
    print(f"Synthetic data saved to {output_csv}")

# Run the function with predefined filenames
generate_synthetic_data(INPUT_CSV, OUTPUT_CSV, NUM_ROWS)