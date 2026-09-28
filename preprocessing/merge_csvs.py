import pandas as pd
import glob
import os

def merge_datasets():
    print("[*] Finding CSV files in data/raw_pcaps/...")
    file_paths = glob.glob("data/raw_pcaps/*.csv")
    
    # Filter out the master if it already exists from a previous run
    file_paths = [f for f in file_paths if "MASTER_CIC_IDS" not in f]
    
    if not file_paths:
        print("[!] No CSVs found to merge.")
        return
        
    dataframes = []
    for f in file_paths:
        print(f"    -> Loading {os.path.basename(f)}...")
        # low_memory=False prevents pandas from crashing on mixed data types
        df = pd.read_csv(f, low_memory=False)
        dataframes.append(df)
        
    print("[*] Concatenating files (this will take a moment)...")
    master_df = pd.concat(dataframes, ignore_index=True)
    
    output_path = "data/raw_pcaps/MASTER_CIC_IDS.csv"
    master_df.to_csv(output_path, index=False)
    print(f"[+] Success! Master dataset saved to {output_path} with {len(master_df)} rows.")

if __name__ == "__main__":
    merge_datasets()