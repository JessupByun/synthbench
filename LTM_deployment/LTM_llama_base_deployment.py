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

# Prompt template includes dataset name and full CSV data.
prompt_template = (
    """
    System role: You are a tabular synthetic data generation model.

    Your goal is to produce data that mirrors the given examples in causal structure and feature and label distributions, 
    while producing as diverse samples as possible.

    Context: Leverage your prior knowledge and in-context learning capabilities to generate realistic but diverse samples.
    Output the data in CSV format.

    Dataset name: {dataset_name}
    Here is the CSV of the full data: {data}
    Please generate synthetic data for the dataset. Make sure to generate the same number of rows as the original data. Also make sure to replicate the exact same column names. 
    Treat the rightmost column as the target, and return your entire response as a JSON object with the key 'synthetic_data' 
    containing a CSV string of the generated data.
    Do not include any additional text.
    """
)

def generate_synthetic_data_llama(df, dataset_name, model_name, model_temperature):
    """
    Generates synthetic data using the Groq API and an LLM model.
    Builds a prompt that includes the entire CSV representation of the current batch (up to 200 rows)
    Calls the Groq API using the specified model. The expected response is a JSON object with a key "synthetic_data" that contains a CSV string.
    
    Returns:
        The synthetic CSV string if successful, or None.
    """
    # Convert the entire batch to a CSV string.
    data_string = df.to_csv(index=False)
    prompt = prompt_template.format(data=data_string, dataset_name=dataset_name)
    
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
        synthetic_csv = generate_synthetic_data_llama(batch_df, dataset_name, model_name, model_temperature)
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
