import os
import sys
import pandas as pd
import numpy as np

def clean_cic_dataset(input_csv_path: str, output_csv_path: str):
    """
    Cleans the official CIC-IDS-2017 Machine Learning CSVs.
    Fixes column names, removes Infinity/NaN values, and preps for the Normalizer.
    """
    print(f"[*] Loading massive CSV file: {input_csv_path}...")
    # Read the CSV (low_memory=False prevents mixed-type warnings on huge files)
    df = pd.read_csv(input_csv_path, low_memory=False)
    
    print(f"[*] Original Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Strip hidden spaces from column names (CIC-IDS is notorious for ' Destination Port')
    df.columns = df.columns.str.strip()
    
    # 2. Drop irrelevant metadata columns that aren't ML features
    cols_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # 3. Fix the Infinity and NaN trap!
    print("[*] Scrubbing NaN and Infinity values...")
    # Replace Inf with NaN, then drop any rows containing NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    # Ensure all feature columns are strictly numeric
    feature_cols = [c for c in df.columns if c != 'Label']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop any new NaNs created by forced numeric conversion
    df.dropna(inplace=True)

    print(f"[*] Cleaned Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Save to the location where our Normalizer expects it
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[+] Cleaned dataset ready for normalization at: {output_csv_path}")

if __name__ == "__main__":
    # Change this to the exact name of the file you downloaded!
    input_file ="data/raw_pcaps/MASTER_CIC_IDS.csv" 
    
    # We output it to the exact spot the Normalizer looks for
    output_file = "data/processed_csv/integration_raw.csv"
    
    if not os.path.exists(input_file):
        print(f"[!] Please place your downloaded CIC-IDS CSV at: {input_file}")
    else:
        clean_cic_dataset(input_file, output_file)