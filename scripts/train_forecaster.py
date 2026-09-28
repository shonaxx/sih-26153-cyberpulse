import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)


# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gru_forecaster import TemporalForecaster

def map_cic_labels_to_mitre(label_series):
    """Maps raw CIC-IDS-2017 string labels to our 0-5 MITRE stages."""
    mapping = {
        'BENIGN': 0,                     
        'PortScan': 1,                   
        'FTP-Patator': 2,                
        'SSH-Patator': 2,                
        r'Web Attack \xbrute force': 2,   
        'Web Attack - Brute Force': 2,
        r'Web Attack \xsqli': 2,          
        'Web Attack - Sql Injection': 2,
        r'Web Attack \xxss': 2,           
        'Web Attack - XSS': 2,
        'Infiltration': 3,               
        'Bot': 4,                        
        'DoS Hulk': 4,                   
        'DoS GoldenEye': 4,
        'DoS slowloris': 4,
        'DoS Slowhttptest': 4,
        'DDoS': 4,
        'Heartbleed': 5                  
    }
    return label_series.map(mapping).fillna(0).astype(int)

def create_sequences_with_labels(features, labels, seq_length, horizons):
    """Slides over the dataset to create 3D input tensors and future target horizons."""
    xs, y_risk, y_stage = [], [], []
    for i in range(len(features) - seq_length - horizons):
        x_seq = features[i:(i + seq_length)]
        future_labels = labels[(i + seq_length):(i + seq_length + horizons)]
        risk_targets = [1.0 if l > 0 else 0.0 for l in future_labels]
        
        xs.append(x_seq)
        y_risk.append(risk_targets)
        y_stage.append(future_labels)
        
    return (
        torch.tensor(np.array(xs, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.array(y_risk, dtype=np.float32), dtype=torch.float32).unsqueeze(-1), 
        torch.tensor(np.array(y_stage, dtype=np.int64), dtype=torch.long)                  
    )

def train_model():
    print("=========================================================")
    print("        TRAINING THE TEMPORAL FORECASTER (BATCHED)       ")
    print("=========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on Hardware: {device.type.upper()}")
    
    data_path = "data/processed_csv/integration_scaled.csv" 
    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    if 'Label' not in df.columns:
        df['Label'] = 'BENIGN'
        
    mitre_labels = map_cic_labels_to_mitre(df['Label']).values
    feature_cols = [c for c in df.columns if c not in ['window_start', 'window_end', 'Label']]
    
    # FORCE numeric conversion to fix the numpy.object_ error
    features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
    
    SEQ_LENGTH = 6
    HORIZONS = 4
    print(f"[*] Windowing data (History={SEQ_LENGTH}, Horizons={HORIZONS})...")
    X_train, y_risk, y_stage = create_sequences_with_labels(features, mitre_labels, SEQ_LENGTH, HORIZONS)
    print(f"[*] Total Tensor Shape: {tuple(X_train.shape)}")
    
    # --- MEMORY-SAFE BATCHING (Fixes the RAM crash) ---
    dataset = TensorDataset(X_train, y_risk, y_stage)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    print(f"[*] Created DataLoader with Batch Size 2048")
    
    model = TemporalForecaster(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_risk = nn.BCELoss()
    criterion_stage = nn.CrossEntropyLoss()
    
    epochs = 10  # Reduced to 10 for faster hackathon training
    print("\n[*] Starting Training Loop...")
    
    model.train()
    for epoch in range(epochs):
        total_loss_accum = 0.0
        
        for batch_idx, (b_x, b_risk, b_stage) in enumerate(dataloader):
            b_x, b_risk, b_stage = b_x.to(device), b_risk.to(device), b_stage.to(device)
            
            optimizer.zero_grad()
            outputs = model(b_x)
            
            loss_risk = criterion_risk(outputs['risk_trajectory'], b_risk)
            logits_flat = outputs['stage_trajectory'].view(-1, 6)
            targets_flat = b_stage.view(-1)
            loss_stage = criterion_stage(logits_flat, targets_flat)
            
            loss = loss_risk + (0.8 * loss_stage) 
            loss.backward()
            optimizer.step()
            
            total_loss_accum += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"    Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Batch Loss: {loss.item():.4f}")
                
        avg_loss = total_loss_accum / len(dataloader)
        print(f"--> END OF EPOCH {epoch+1} | Average Loss: {avg_loss:.4f}\n")
        
    save_path = "models/saved_weights/gru_forecaster.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Master Model weights saved successfully to {save_path}")

if __name__ == "__main__":
    train_model()