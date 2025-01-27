from evaluator import *
import pandas as pd
import os

# Using the automated EvaluationPipeline
# Currently adapted to: private_combined_df_cty_week2_for_synth.csv

"""
real_data: A csv file containing the real data
synth_data: A csv file containing the synthetic data
column_name_to_datatype: A dictionary mapping each column name to a string representing its datatype (see next section for format)
config: A dictionary containing the configuration for the evaluation pipeline (see next section for format)
save_path: A string representing the path to a directory to save the tables and plots (MUST LEAD TO A DIRECTORY)
"""

# Load the real and synthetic data
real_data = pd.read_csv("data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_for_synth.csv")
real_test_data = pd.read_csv("data/real_data/private_combined_df_cty_week2/private_combined_df_cty_week2_test.csv")
synth_data_path = "data/synthetic_data/private_combined_df_cty_week2/combined_df_cty_week2_synthetic_data_llama70B_n200_temp1.0_advanced_prompt.csv"
synth_data = pd.read_csv(synth_data_path)

# Extract the base filename (without directory path) for use in the save path
base_filename = os.path.basename(synth_data_path)
save_path = f"alfred_evaluation_{base_filename}"

column_name_to_datatype = {
    "state": "categorical",
    "county_fips": "categorical",
    "week": "categorical",
    "mask_user_pct": "numerical",
    "mask_mandate": "categorical",
    "gop_vote_share_2016": "numerical",
    "deaths_per_10k": "numerical",
    "COVID_news": "numerical",
    "retail_visit_per_hundred": "numerical",
    "COVID_news_cable": "numerical",
    "urban_population_percentage": "numerical",
    "image_users": "numerical",
    "mask_users": "numerical",
    "population_density": "numerical",
    "week_counter": "numerical",
    "week_counter_log": "numerical",
    "population": "numerical"
}

config = {
    "target_column": "mask_user_pct",
    "fidelity_metrics": ["SumStats", "ColumnShape"],
    "holdout_seed": 42,
    "holdout_size": 0.2,
    "holdout_index" : list(real_test_data.index),
    "metadata": {
         "state": "categorical",
        "county_fips": "categorical",
        "week": "categorical",
        "mask_user_pct": "numerical",
        "mask_mandate": "categorical",
        "gop_vote_share_2016": "numerical",
        "deaths_per_10k": "numerical",
        "COVID_news": "numerical",
        "retail_visit_per_hundred": "numerical",
        "COVID_news_cable": "numerical",
        "urban_population_percentage": "numerical",
        "image_users": "numerical",
        "mask_users": "numerical",
        "population_density": "numerical",
        "week_counter": "numerical",
        "week_counter_log": "numerical",
        "population": "numerical"
    }
}

# Run the evaluation pipeline
evaluation_pipeline = EvaluationPipeline(real_data=real_data, synth_data=synth_data, column_name_to_datatype=column_name_to_datatype, config=config, save_path=save_path)
evaluation_pipeline.run_pipeline()
