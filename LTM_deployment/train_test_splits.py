import os
import pandas as pd
import numpy as np
import yaml
import json
from sklearn.model_selection import train_test_split
import random
import glob
from pathlib import Path

# Constants
TEST_RATIO = 0.2
RANDOM_STATE = 42
SUBSET_SIZES = [8, 16, 32, 64, 128]
NUM_SUBSETS = 3
FULL_SIZE_CODE = 999  # Code for full train/test sets

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "TrainTestSplits")

# Directories to search for datasets
DATASET_DIRS = [
    os.path.join(ROOT_DIR, "openml_ctr23"),
    os.path.join(ROOT_DIR, "grinsztajn"),
    os.path.join(ROOT_DIR, "openml_cc18"),
    os.path.join(ROOT_DIR, "unipredict")
]

def find_leaf_folders(root_dir):
    """Find all leaf folders (folders with no subfolders) that contain required files in a specific root directory."""
    leaf_folders = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # If there are no subdirectories, this is a leaf folder
        if not dirnames:
            # Check if the folder contains the required files
            csv_files = [f for f in filenames if f.endswith('.csv')]
            yaml_files = [f for f in filenames if f.endswith('.yaml') or f.endswith('.yml')]
            jsonl_files = [f for f in filenames if f.endswith('.jsonl')]
            
            if csv_files and yaml_files and jsonl_files:
                leaf_folders.append({
                    'path': dirpath,
                    'csv_file': os.path.join(dirpath, csv_files[0]),
                    'yaml_file': os.path.join(dirpath, yaml_files[0]),
                    'jsonl_file': os.path.join(dirpath, jsonl_files[0]),
                    'name': os.path.basename(dirpath).replace('_', '-'),
                    'source_dir': os.path.basename(root_dir)
                })
    
    return leaf_folders

def process_dataset(dataset_info, source_output_dir):
    """Process a single dataset: split into train/test and create subsets."""
    dataset_name = dataset_info['name']
    print(f"Processing dataset: {dataset_name}")
    
    # Create dataset-specific folder with train and test subfolders
    dataset_dir = os.path.join(source_output_dir, dataset_name)
    dataset_train_dir = os.path.join(dataset_dir, "train")
    dataset_test_dir = os.path.join(dataset_dir, "test")
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(dataset_train_dir, exist_ok=True)
    os.makedirs(dataset_test_dir, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(dataset_info['csv_file'])
    
    # Split into train and test sets
    X = df.iloc[:, :-1]  # All columns except the last one (assuming last column is target)
    y = df.iloc[:, -1]   # Last column as target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    
    # Recombine features and target
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save the full test set
    test_filename = f"{dataset_name}--test--{FULL_SIZE_CODE}.csv"
    test_path = os.path.join(dataset_test_dir, test_filename)
    test_df.to_csv(test_path, index=False)
    print(f"  Saved test set: {test_filename}")
    
    # Save the full training set
    train_filename = f"{dataset_name}--train--{FULL_SIZE_CODE}.csv"
    train_path = os.path.join(dataset_train_dir, train_filename)
    train_df.to_csv(train_path, index=False)
    print(f"  Saved full training set: {train_filename}")
    
    # Create and save subsets of different sizes
    for size in SUBSET_SIZES:
        # Skip if the training set is smaller than the requested size
        if len(train_df) < size:
            print(f"  Skipping size {size} (training set too small)")
            continue
        
        for seed in range(NUM_SUBSETS):
            # Sample the subset
            subset = train_df.sample(n=size, random_state=seed)
            
            # Save the subset
            subset_filename = f"{dataset_name}--train--{size}-seed{seed}.csv"
            subset_path = os.path.join(dataset_train_dir, subset_filename)
            subset.to_csv(subset_path, index=False)
            print(f"  Saved subset: {subset_filename}")

def main():
    """Main function to process all datasets."""
    print("Finding leaf folders...")
    
    # Process each dataset directory separately
    for dataset_dir in DATASET_DIRS:
        source_dir_name = os.path.basename(dataset_dir)
        print(f"\nProcessing source directory: {source_dir_name}")
        
        # Create source-specific output directory
        source_output_dir = os.path.join(OUTPUT_DIR, source_dir_name)
        os.makedirs(source_output_dir, exist_ok=True)
        
        # Find leaf folders for this source directory
        leaf_folders = find_leaf_folders(dataset_dir)
        print(f"Found {len(leaf_folders)} leaf folders with required files in {source_dir_name}.")
        
        # Process each dataset in this source directory
        for dataset_info in leaf_folders:
            process_dataset(dataset_info, source_output_dir)
        
        break
    
    print("\nAll datasets processed successfully!")

if __name__ == "__main__":
    main()
