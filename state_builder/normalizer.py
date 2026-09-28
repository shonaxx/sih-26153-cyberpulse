import os
import sys
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StateNormalizer:
    """
    Scales network traffic features to handle massive outliers (like bandwidth spikes)
    without corrupting the neural network gradients.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_cols = []

    def fit_transform_csv(self, input_csv: str, output_csv: str, scaler_save_path: str):
        df = pd.read_csv(input_csv, low_memory=False)
        
        # 1. Isolate string labels and metadata so we don't feed text to the math scaler
        label_col = None
        if 'Label' in df.columns:
            label_col = df['Label']
            df = df.drop(columns=['Label'])
            
        meta_cols = {}
        for col in ['window_start', 'window_end']:
            if col in df.columns:
                meta_cols[col] = df[col]
                df = df.drop(columns=[col])
                
        self.feature_cols = df.columns.tolist()
        
        # 2. Scale the pure numerical features
        scaled_features = self.scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_cols)
        
        # 3. Reattach the metadata and text labels
        for col, data in meta_cols.items():
            scaled_df[col] = data
        if label_col is not None:
            scaled_df['Label'] = label_col
            
        # 4. Save outputs
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        scaled_df.to_csv(output_csv, index=False)
        
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_save_path)

    def transform_live_state(self, state_dict: dict) -> dict:
        # (Used later for the live dashboard)
        pass

if __name__ == "__main__":
    print("--- Normalizing CIC-IDS Dataset ---")
    
    input_csv = "data/processed_csv/integration_raw.csv"
    output_csv = "data/processed_csv/integration_scaled.csv"
    scaler_out = "data/scalers/integration_scaler.pkl"
    
    if not os.path.exists(input_csv):
        print(f"[!] Cannot find {input_csv}. Did you run the ingestor script first?")
    else:
        print(f"[*] Loading raw training data from {input_csv}...")
        print("[*] Fitting RobustScaler to the numerical data (ignoring text labels)...")
        normalizer = StateNormalizer()
        normalizer.fit_transform_csv(input_csv, output_csv, scaler_out)
        print("[+] Massive dataset normalized successfully!")