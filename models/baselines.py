# Logistic Regression and Random Forest static classifiers
import os
import sys
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StaticBaselineModels:
    """
    Standard machine learning models used to prove the necessity of temporal forecasting.
    These look at a single time window X[t] and classify it (Current State only).
    """
    def __init__(self):
        # max_iter=1000 ensures it converges even on messy network data
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        # n_estimators=50 keeps it lightweight for your laptop
        self.rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Trains the baselines and prints the SIH evaluation metrics."""
        print(f"[*] Splitting {len(X)} records into Train/Test sets...")
        
        # 70% Training / 30% Testing split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 1. Train and Evaluate Logistic Regression
        print("[*] Training Logistic Regression...")
        self.lr_model.fit(X_train, y_train)
        lr_preds = self.lr_model.predict(X_test)
        
        print("\n=========================================")
        print("  LOGISTIC REGRESSION (STATIC BASELINE)  ")
        print("=========================================")
        print(classification_report(y_test, lr_preds, zero_division=0))
        
        # 2. Train and Evaluate Random Forest
        print("\n[*] Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        rf_preds = self.rf_model.predict(X_test)
        
        print("\n=========================================")
        print("    RANDOM FOREST (STATIC BASELINE)      ")
        print("=========================================")
        print(classification_report(y_test, rf_preds, zero_division=0))

    def save_models(self, save_dir: str):
        """Saves the trained models for the dashboard comparison."""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "logistic_regression.pkl"), "wb") as f:
            pickle.dump(self.lr_model, f)
            
        with open(os.path.join(save_dir, "random_forest.pkl"), "wb") as f:
            pickle.dump(self.rf_model, f)
            
        print(f"\n[+] Baseline models saved successfully to {save_dir}/")

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    import numpy as np
    
    print("--- Testing ML Baseline Training ---")
    
    # Let's generate a slightly larger synthetic dataset (100 rows) to test the ML math
    # We will simulate 70 normal windows (Label 0) and 30 attack windows (Label 1)
    
    # 70 Normal Windows: Low entropy, low SYN rate
    normal_data = pd.DataFrame({
        "total_bytes": np.random.uniform(100, 500, 70),
        "syn_rate": np.random.uniform(0.1, 2.0, 70),
        "dst_port_entropy": np.random.uniform(0.0, 0.3, 70)
    })
    normal_labels = pd.Series(np.zeros(70))
    
    # 30 Attack Windows (Port Scans): High entropy, high SYN rate
    attack_data = pd.DataFrame({
        "total_bytes": np.random.uniform(500, 2000, 30),
        "syn_rate": np.random.uniform(20.0, 100.0, 30),
        "dst_port_entropy": np.random.uniform(0.8, 1.0, 30)
    })
    attack_labels = pd.Series(np.ones(30))
    
    # Combine them
    X_synthetic = pd.concat([normal_data, attack_data], ignore_index=True)
    y_synthetic = pd.concat([normal_labels, attack_labels], ignore_index=True)
    
    # Run the models!
    baselines = StaticBaselineModels()
    baselines.train_and_evaluate(X_synthetic, y_synthetic)
    baselines.save_models("models/saved_weights")