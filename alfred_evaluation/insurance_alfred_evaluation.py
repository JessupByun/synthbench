from evaluator import *
import pandas as pd
import os

# Using the automated EvaluationPipeline
# Currently adapted to: insurance.csv

"""
real_data: A csv file containing the real data
synth_data: A csv file containing the synthetic data
column_name_to_datatype: A dictionary mapping each column name to a string representing its datatype (see next section for format)
config: A dictionary containing the configuration for the evaluation pipeline (see next section for format)
save_path: A string representing the path to a directory to save the tables and plots (MUST LEAD TO A DIRECTORY)
"""

# Load the real and synthetic data
real_data = pd.read_csv("data/real_data/insurance/insurance.csv")
real_test_data = pd.read_csv("data/real_data/insurance/insurance_test.csv")
synth_data_path = "data/synthetic_data/insurance/insurance_synthetic_data_llama70B_n250_temp1.0_advanced_prompt.csv"
synth_data = pd.read_csv(synth_data_path)

# Extract the base filename (without directory path) for use in the save path
base_filename = os.path.basename(synth_data_path)
save_path = f"alfred_evaluation_{base_filename}"

column_name_to_datatype = {
    "age": "numerical",
    "sex": "categorical",
    "bmi": "numerical",
    "children": "categorical",
    "smoker": "categorical",
    "region": "categorical",
    "charges": "numerical"
}

config = {
    "target_column": "charges",
    "fidelity_metrics": ["SumStats", "ColumnShape"],
    "holdout_seed": 42,
    "holdout_size": 0.2,
    "holdout_index" : list(real_test_data.index),
    "metadata": {
        "age": "numerical",
        "sex": "categorical",
        "bmi": "numerical",
        "children": "categorical",
        "smoker": "categorical",
        "region": "categorical",
        "charges": "numerical"
    }
}

# Run the evaluation pipeline
evaluation_pipeline = EvaluationPipeline(real_data=real_data, synth_data=synth_data, column_name_to_datatype=column_name_to_datatype, config=config, save_path=save_path)
evaluation_pipeline.run_pipeline()
