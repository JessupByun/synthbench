from evaluator import *
import pandas as pd
import os

# Using the automated EvaluationPipeline
# Currently adapted to: labMergeDemo1Trimmed.csv

"""
real_data: A csv file containing the real data
synth_data: A csv file containing the synthetic data
column_name_to_datatype: A dictionary mapping each column name to a string representing its datatype (see next section for format)
config: A dictionary containing the configuration for the evaluation pipeline (see next section for format)
save_path: A string representing the path to a directory to save the tables and plots (MUST LEAD TO A DIRECTORY)
"""

# Load the real and synthetic data
real_data = pd.read_csv("data/real_data/private_labMergeDemo1Trimmed/private_labMergeDemo1Trimmed.csv")
real_test_data = pd.read_csv("data/real_data/private_labMergeDemo1Trimmed/private_labMergeDemo1Trimmed_test.csv")
synth_data_path = "data/synthetic_data/private_labMergeDemo1Trimmed/labMergeDemo1Trimmed_synthetic_data_llama70B_n250_temp1.0_advanced_prompt.csv"
synth_data = pd.read_csv(synth_data_path)

# Extract the base filename (without directory path) for use in the save path
base_filename = os.path.basename(synth_data_path)
save_path = f"alfred_evaluation_{base_filename}"

column_name_to_datatype = {
    "subject_id_x": "numerical",
    "admittime_y": "categorical",
    "dischtime": "categorical",
    "Age": "numerical",
    "gender": "categorical",
    "ethnicity": "categorical",
    "insurance": "categorical",
    "label": "numerical",
    "dod": "categorical",
    "charttime": "categorical",
    "admittime_x": "categorical",
    "lab_time_from_admit": "categorical",
    "valuenum": "numerical"
}

config = {
    "target_column": "valuenum",
    "fidelity_metrics": ["SumStats", "ColumnShape"],
    "holdout_seed": 42,
    "holdout_size": 0.2,
    "holdout_index" : list(real_test_data.index),
    "metadata": {
        "subject_id_x": "numerical",
        "admittime_y": "categorical",
        "dischtime": "categorical",
        "Age": "numerical",
        "gender": "categorical",
        "ethnicity": "categorical",
        "insurance": "categorical",
        "label": "numerical",
        "dod": "categorical",
        "charttime": "categorical",
        "admittime_x": "categorical",
        "lab_time_from_admit": "categorical",
        "valuenum": "numerical"
    }
}

# Run the evaluation pipeline
evaluation_pipeline = EvaluationPipeline(real_data=real_data, synth_data=synth_data, column_name_to_datatype=column_name_to_datatype, config=config, save_path=save_path)
evaluation_pipeline.run_pipeline()
