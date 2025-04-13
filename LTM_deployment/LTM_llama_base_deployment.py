import os
import json
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import groq

# Retrieve the Groq API key and instantiate client
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")
client = groq.Groq(api_key=api_key)

# Prompt template includes dataset name, summary statistcs, column names, and full CSV data.
prompt_template = (
    """
    System role: You are a tabular synthetic data generation model.

    Your goal is to produce data that mirrors the given examples in causal structure and feature and label distributions, while producing as diverse samples as possible.

    Context: Leverage your prior knowledge and in-context learning capabilities to generate realistic but diverse samples.
    Output the data in CSV format.

    Dataset name: {dataset_name}
    Column names: {col_names}
    Summary statistics and information about numerical and categorical columns: {summary_stats}
    Here is the CSV of the full data: {data}
    Please generate {batch_size} rows of synthetic data for the dataset. 

    Treat the rightmost column as the target, and return your entire response as a JSON object with the key 'synthetic_data' containing a CSV string of the generated data.
    Do not include any additional text.
    """
)

def get_summary_statistics(df):
    """
    Computes a comprehensive set of summary statistics for each column in the DataFrame.
    
    For numeric columns, it calculates:
      - mean, median, mode (first mode value if multiple), standard deviation, min, max,
      - 25th and 75th percentiles,
      - number of unique values.
      
    For non-numeric (categorical) columns, it calculates:
      - the number of unique values,
      - the most common value (mode),
      - and the full value counts as a dictionary.
    
    Returns:
        A JSON string representation of the summary statistics.
    """
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate basic statistics for numeric columns.
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "mode": float(mode_val) if pd.notnull(mode_val) else None,
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "25%": float(df[col].quantile(0.25)),
                "75%": float(df[col].quantile(0.75)),
                "unique_count": int(df[col].nunique())
            }
        else:
            # For non-numeric columns, provide the number of unique values, the mode, and all value counts.
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            value_counts = df[col].value_counts().to_dict()
            # Convert keys to strings in case the non-numeric values are not string type.
            value_counts = {str(k): int(v) for k, v in value_counts.items()}
            stats[col] = {
                "unique_count": int(df[col].nunique()),
                "mode": str(mode_val) if mode_val is not None else None,
                "value_counts": value_counts
            }
    return json.dumps(stats, indent=2)

def generate_synthetic_data_llama(df, dataset_name, model_name, batch_size, model_temperature):
    """
    Generates synthetic data using the Groq API and an LLM model.
    
    This function builds a prompt that includes:
      - The dataset name
      - The full list of column names
      - The summary statistics for the DataFrame (numeric columns)
      - The entire CSV representation of the current batch (up to 200 rows)
    It then calls the Groq API using the specified model and expects a JSON object with the key 
    "synthetic_data" that contains the synthetic CSV string.
    
    Returns:
        The synthetic CSV string if successful, or None.
    """
    # Convert the entire batch to a CSV string.
    data_string = df.to_csv(index=False)
    
    # Get column names and summary statistics.
    col_names = ", ".join(df.columns)
    summary_stats = get_summary_statistics(df)

    # Build the prompt using the template.
    prompt = prompt_template.format(
        data=data_string, 
        dataset_name=dataset_name,
        col_names=col_names,
        summary_stats=summary_stats,
        batch_size = batch_size
    )

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            response_format={"type": "json_object"},  # Enable JSON output mode.
            temperature=model_temperature
        )
        #print("Full Response:", response)
        
        if response.choices and len(response.choices) > 0:
            generated_text = response.choices[0].message.content
        else:
            print("No choices were returned in the response.")
            return None

        parsed = json.loads(generated_text)
        if "synthetic_data" in parsed:
            return parsed["synthetic_data"]
        else:
            print("Returned JSON does not contain 'synthetic_data' key.")
            return None
    except Exception as e:
        print(f"Error generating data with model {model_name}: {e}")
        return None

def process_csv_file_llama(input_csv, output_csv, dataset_name, model_name, model_temperature, batch_size=200):
    """
    Loads a CSV file and shuffles it (using seed=42). If it has more than batch_size rows, partitions it into chunks.
    For each chunk, the entire batch is converted to CSV and synthetic data is generated via generate_synthetic_data_llama.
    The resulting synthetic DataFrames are concatenated and saved to output_csv.
    """
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_rows = df.shape[0]
    
    synthetic_df_list = []
    for start in range(0, n_rows, batch_size):
        batch_df = df.iloc[start:start+batch_size]
        synthetic_csv = generate_synthetic_data_llama(batch_df, dataset_name, model_name, batch_size, model_temperature)
        if synthetic_csv is None:
            print(f"No synthetic data returned for batch starting at row {start}.")
            continue
        try:
            batch_synthetic_df = pd.read_csv(StringIO(synthetic_csv))
        except Exception as e:
            print(f"Error converting synthetic CSV to DataFrame for batch starting at row {start}: {e}")
            continue
        synthetic_df_list.append(batch_synthetic_df)
    
    if len(synthetic_df_list) == 0:
        print("No synthetic data generated for file:", input_csv)
        return
    
    synthetic_df = pd.concat(synthetic_df_list, ignore_index=True)
    if synthetic_df.shape[0] > n_rows:
        synthetic_df = synthetic_df.iloc[:n_rows, :]
    if synthetic_df.shape[0] != n_rows:
        print(f"[WARNING] Synthetic row count ({synthetic_df.shape[0]}) differs from original ({n_rows}).")
    
    synthetic_df.to_csv(output_csv, index=False)
    print(f"[INFO] Synthetic data saved to {output_csv}")

def process_dataset_llama(dataset_name, generator_name, model_name, model_temperature, batch_size=200):
    """
    Processes all CSV files found in:
      LTM_data/LTM_real_data/{dataset_name}/train/
    Synthetic CSVs are saved under:
      LTM_data/LTM_synthetic_data/LTM_llama_synthetic_data/synth_{dataset_name}/
    Each output file is named:
      {original_csv_filename}_{generator_name}_default_0.csv

    After processing, validation is run using the provided validation script.
    """
    real_data_path = os.path.join("LTM_data", "LTM_real_data", dataset_name)
    train_folder = os.path.join(real_data_path, "train")
    if not os.path.isdir(train_folder):
        print(f"[ERROR] Train folder not found: {train_folder}")
        return
    
    synthetic_folder = os.path.join("LTM_data", "LTM_synthetic_data", "LTM_llama_synthetic_data", f"synth_{dataset_name}")
    os.makedirs(synthetic_folder, exist_ok=True)
    
    csv_files = [f for f in os.listdir(train_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"[WARNING] No CSV files found in {train_folder}.")
        return
    
    for csv_file in csv_files:
        input_csv_path = os.path.join(train_folder, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        output_csv_name = f"{base_name}_{generator_name}_default_0.csv"
        output_csv_path = os.path.join(synthetic_folder, output_csv_name)
        print(f"[INFO] Processing: {input_csv_path} -> {output_csv_path}")
        process_csv_file_llama(input_csv_path, output_csv_path, dataset_name, model_name, model_temperature, batch_size=batch_size)
    
    # Run validation using the provided validation script's structure.
    try:
        from validate_synthetic_data import validate_synthetic_data, logger
    except ImportError as e:
        print(f"Error importing validation function: {e}")
        return
    
    # Build arguments as if they were parsed.
    args = {
        "real_data_dir": os.path.join("LTM_data", "LTM_real_data", dataset_name, "train"),
        "synthetic_data_dir": synthetic_folder,
        "output_file": os.path.join(synthetic_folder, f"{dataset_name}_validation_results.json")
    }
    
    validation_results = validate_synthetic_data(args["real_data_dir"], args["synthetic_data_dir"])
    
    # Print summary.
    passed = sum(1 for result in validation_results if result["validation_passed"])
    total = len(validation_results)
    logger.info(f"Validation complete: {passed}/{total} synthetic datasets passed all checks")
    
    for result in validation_results:
        if not result["validation_passed"]:
            logger.warning(f"Issues with {result['synthetic_file']}:")
            for issue in result["issues"]:
                logger.warning(f"  - {issue}")
    
    # Save validation results.
    try:
        with open(args["output_file"], "w") as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {args['output_file']}")
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")

def main():
    dataset_name = "abalone"
    generator_name = "llama"
    model_name = "llama-3.3-70b-versatile"
    model_temperature = 1.0 # Leave as 1.0 for highest diversity
    batch_size = 200

    process_dataset_llama(dataset_name, generator_name=generator_name, model_name=model_name, model_temperature=model_temperature, batch_size=batch_size)

if __name__ == "__main__":
    main()
