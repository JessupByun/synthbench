#!/usr/bin/env python3
"""
llama_ablation_synth.py

Generate synthetic data with Groq+Llama for your ablation studies,
handling batch splits, JSON parsing, and validation exactly as before,
but now automatically dropping any spurious header rows in each batch.

Usage:
  1. Set GROQ_API_KEY in your .env.
  2. Edit `main()` for dataset_name, model_name, temperature, batch_size.
  3. Run: python llama_ablation_synth.py
"""
import os
import re
import json
import time
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import groq

# —— CONFIGURATION ——
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")
client = groq.Groq(api_key=api_key)

prompt_template = (
    """
    System role: You are a tabular synthetic data generation model.

    Your goal is to produce data that mirrors the given examples in causal structure
    and feature and label distributions, while producing as diverse samples as possible.

    Context: Leverage your prior knowledge and in-context learning capabilities
    to generate realistic but diverse samples. Output your result as JSON.

    Dataset name: {dataset_name}
    Columns in order: {col_names}
    Enclose any non-numeric cell in double-quotes.
    Do not emit trailing commas or extra columns.
    Summary stats: {summary_stats}
    Here is the CSV of the full data:
    {data}

    Generate exactly {batch_size} rows of synthetic data.
    Return a JSON object with key "synthetic_data" whose value is the CSV text.
    No extra text.
    """
)
# ——————————————————

def get_summary_statistics(df):
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mode = df[col].mode()
            mode = float(mode.iloc[0]) if not mode.empty else None
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "mode": mode,
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "25%": float(df[col].quantile(0.25)),
                "75%": float(df[col].quantile(0.75)),
                "unique_count": int(df[col].nunique())
            }
        else:
            mode = df[col].mode()
            mode = str(mode.iloc[0]) if not mode.empty else None
            vc = df[col].value_counts().to_dict()
            stats[col] = {
                "unique_count": int(df[col].nunique()),
                "mode": mode,
                "value_counts": {str(k): int(v) for k,v in vc.items()}
            }
    return json.dumps(stats)

def extract_required_n_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if "--train--" not in base:
        return None
    part = base.split("--train--",1)[1]
    num = part.split("-seed",1)[0] if "-seed" in part else part
    try:
        return int(num)
    except:
        return None

def generate_synthetic_data_llama(df, dataset_name, model_name, batch_size, temperature):
    data_csv = df.to_csv(index=False)
    col_names = ", ".join(df.columns)
    summary_json = get_summary_statistics(df)

    prompt = prompt_template.format(
        data=data_csv,
        dataset_name=dataset_name,
        col_names=col_names,
        summary_stats=summary_json,
        batch_size=batch_size
    )

    try:
        resp = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model=model_name,
            temperature=temperature
        )
        raw = resp.choices[0].message.content

        # strip markdown fences
        raw = re.sub(r"^```[^\n]*\n", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

        # extract JSON object substring
        start = raw.find("{")
        end   = raw.rfind("}")
        json_str = raw[start:end+1] if start>=0 and end>=0 else raw
        parsed = json.loads(json_str)
        return parsed.get("synthetic_data", None)
    except Exception as e:
        print(f"[ERROR] generating batch: {e}")
        return None

def drop_header_rows(df):
    """
    Drop any rows that exactly match the column names (spurious headers).
    """
    # build a Series of column names for comparison
    col_series = pd.Series(df.columns, index=df.columns)
    # mask rows where every cell equals its column name
    mask = (df == col_series).all(axis=1)
    if mask.any():
        df = df.loc[~mask].reset_index(drop=True)
    return df

def process_csv_file_llama(input_csv, output_csv, dataset_name, model_name, temperature, batch_size=200):
    df = pd.read_csv(input_csv).sample(frac=1, random_state=42).reset_index(drop=True)
    required_n = extract_required_n_from_filename(input_csv) or len(df)

    synthetic_chunks = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start+batch_size]
        while True:
            csv_text = generate_synthetic_data_llama(batch, dataset_name, model_name, batch_size, temperature)
            if not csv_text:
                print(f"[WARN] retry batch start {start}")
                time.sleep(1)
                continue
            try:
                gen_df = pd.read_csv(StringIO(csv_text))
            except Exception as e:
                print(f"[ERROR] parse CSV batch {start}: {e}, retry")
                time.sleep(1)
                continue

            # drop any inserted header rows
            gen_df = drop_header_rows(gen_df)
            if gen_df.empty:
                print(f"[WARN] empty batch {start} after header drop, retry")
                time.sleep(1)
                continue

            synthetic_chunks.append(gen_df)
            break

    if not synthetic_chunks:
        print(f"[ERROR] no synthetic for {input_csv}")
        return

    synth = pd.concat(synthetic_chunks, ignore_index=True)

    # reprompt loop if under
    attempts = 0
    while len(synth) < required_n and attempts < 5:
        needed = required_n - len(synth)
        extra = generate_synthetic_data_llama(df, dataset_name, model_name, min(needed,batch_size), temperature)
        if not extra:
            break
        try:
            extra_df = pd.read_csv(StringIO(extra))
            extra_df = drop_header_rows(extra_df)
            synth = pd.concat([synth, extra_df], ignore_index=True)
        except:
            break
        attempts += 1

    synth = synth.iloc[:required_n]
    if len(synth) != required_n:
        print(f"[WARNING] got {len(synth)} rows vs expected {required_n}")

    synth.to_csv(output_csv, index=False)
    print(f"[INFO] saved synthetic to {output_csv}")

def process_dataset_llama(dataset_name, generator_name, model_name, temperature, batch_size=200):
    real_root = os.path.join("LTM_data","LTM_real_data",dataset_name,"train")
    if not os.path.isdir(real_root):
        print(f"[ERROR] no train folder: {real_root}")
        return
    synth_root = os.path.join("LTM_data","LTM_synthetic_data","LTM_llama_synthetic_data",f"synth_{dataset_name}")
    os.makedirs(synth_root, exist_ok=True)

    for file in os.listdir(real_root):
        if not file.lower().endswith(".csv"):
            continue
        inp = os.path.join(real_root, file)
        base = os.path.splitext(file)[0]
        out = os.path.join(synth_root, f"{base}_{generator_name}_default_0.csv")
        print(f"[INFO] {inp} → {out}")
        process_csv_file_llama(inp, out, dataset_name, model_name, temperature, batch_size)

    # run validation
    try:
        from validate_synthetic_data import validate_synthetic_data, logger
    except ImportError as e:
        print(f"[ERROR] import validate: {e}")
        return

    results = validate_synthetic_data(os.path.join(real_root), synth_root)
    passed = sum(r["validation_passed"] for r in results)
    logger.info(f"Validation: {passed}/{len(results)} passed")
    with open(os.path.join(synth_root, f"{dataset_name}_validation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

def main():
    dataset_name      = "diamonds"
    generator_name    = "llama"
    model_name        = "llama-3.3-70b-versatile"
    model_temperature = 1.0
    batch_size        = 10

    process_dataset_llama(
        dataset_name,
        generator_name,
        model_name,
        model_temperature,
        batch_size
    )

if __name__ == "__main__":
    main()
