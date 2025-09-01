import os
import pandas as pd
import numpy as np
from natsort import natsorted # To sort strings in a natural, human-friendly order

# --- CONFIGURATION ---
BASE_DIR = 'checkpoints_cv2/unet'

# --- SCRIPT LOGIC ---
def process_experiment_results(base_dir):
    """
    Traverses the experiment directories, finds cv_results.csv,
    calculates std dev, and prints the mean and std for each experiment.
    """
    # Check if the base directory exists
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found at '{base_dir}'")
        return

    # Get a naturally sorted list of experiment directories
    try:
        experiment_dirs = natsorted([
            d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))
        ])
    except FileNotFoundError:
        print(f"Error: Could not list directories in '{base_dir}'. It might be empty or permissions are wrong.")
        return

    print("--- PARSING EXPERIMENT RESULTS ---")

    for exp_dir in experiment_dirs:
        # Construct the full path to the results file
        results_path = os.path.join(base_dir, exp_dir, 'cv_results.csv')
        
        if os.path.exists(results_path):
            try:
                # Read the CSV file, setting the first column as the index
                df = pd.read_csv(results_path, index_col=0)
                
                # --- DATA CLEANING AND CALCULATION ---
                # Drop the 'mean' row if it exists, to calculate std on raw fold data
                df_folds = df.drop('mean', errors='ignore')

                # Ensure all data is numeric, coercing errors to NaN
                df_folds = df_folds.apply(pd.to_numeric, errors='coerce')

                # Calculate standard deviation, ignoring NaN values
                std_dev = df_folds.std(axis=0, skipna=True)
                std_dev.name = 'std' # Name the new Series for later use
                
                # --- EXTRACTING FINAL RESULTS ---
                # Get the original mean row from the file
                mean_row = df.loc['mean']
                
                # Print the results in a structured format
                print(f"\n# --- Experiment: {exp_dir} ---")
                
                # Print Mean
                print("mean, ", end="")
                print(",".join([f"{val:.8f}" for val in mean_row.values]))
                
                # Print Standard Deviation
                print("std,  ", end="")
                print(",".join([f"{val:.8f}" for val in std_dev.values]))
                
            except Exception as e:
                print(f"\n# --- Could not process: {exp_dir} ---")
                print(f"  Error: {e}")
        else:
            # This part is commented out to avoid clutter, but can be useful for debugging
            # print(f"\n# --- Skipping: {exp_dir} (cv_results.csv not found) ---")
            pass

if __name__ == '__main__':
    process_experiment_results(BASE_DIR)