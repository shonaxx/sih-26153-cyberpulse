# End-to-end execution script (PCAP -> Dashboard)
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our entire SIH26153 ecosystem
from preprocessing.pcap_parser import stream_pcap
from state_builder.sliding_window import SlidingWindowBuilder
from state_builder.normalizer import StateNormalizer
from models.gru_forecaster import TemporalForecaster
from mitre.mapper import MitreMapper
from explainability.gradient_attribution import FastGradientExplainer

def generate_integration_pcap(pcap_path: str):
    """Generates a 40-second PCAP to ensure we fill the 6-window GRU sequence."""
    from scapy.all import wrpcap, Ether, IP, TCP
    print(f"[*] Generating 40-second integration PCAP at {pcap_path}...")
    
    packets = []
    base_time = time.time()
    # Create packets spanning 40 seconds
    for sec in range(40):
        # Normal traffic for first 20s, then an "attack" starts
        if sec < 20:
            pkt = Ether()/IP(dst="192.168.1.10")/TCP(dport=80, flags="A")
        else:
            # Attack: High entropy scanning
            pkt = Ether()/IP(dst="192.168.1.10")/TCP(dport=1000+sec, flags="S")
            
        pkt.time = base_time + sec
        packets.append(pkt)
        
    os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
    wrpcap(pcap_path, packets)

def run_end_to_end():
    print("=========================================================")
    print("   SIH26153: END-TO-END PIPELINE INTEGRATION TEST        ")
    print("=========================================================\n")

    # 1. Setup Paths
    pcap_file = "data/raw_pcaps/integration_test.pcap"
    raw_csv = "data/processed_csv/integration_raw.csv"
    scaled_csv = "data/processed_csv/integration_scaled.csv"
    scaler_path = "data/scalers/integration_scaler.pkl"

    # Generate test data
    generate_integration_pcap(pcap_file)

    # 2. Run Data Pipeline (State Builder)
    print("\n[PIPELINE STAGE 1] Network State Construction")
    builder = SlidingWindowBuilder(window_size_sec=10.0, step_size_sec=5.0)
    builder.process_pcap_to_csv(pcap_file, raw_csv)

    # 3. Run Normalizer
    print("\n[PIPELINE STAGE 2] Feature Normalization")
    normalizer = StateNormalizer()
    normalizer.fit_transform_csv(raw_csv, scaled_csv, scaler_path)

    # 4. Prepare Tensor Sequence for GRU (W=6)
    print("\n[PIPELINE STAGE 3] Sequence Assembly")
    df = pd.read_csv(scaled_csv)
    
    # Drop timestamp metadata to isolate pure features
    feature_cols = [c for c in df.columns if c not in ['window_start', 'window_end']]
    features_only = df[feature_cols].values
    
    # We need exactly 6 windows for a full sequence. If we have more, we take the latest 6.
    # If we have less, we pad with zeros (though our 40s PCAP guarantees we have enough).
    if len(features_only) >= 6:
        sequence_data = features_only[-6:]
    else:
        pad = np.zeros((6 - len(features_only), len(feature_cols)))
        sequence_data = np.vstack([pad, features_only])
        
    # Convert to PyTorch Tensor: Shape (Batch=1, SeqLen=6, Features)
    seq_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
    print(f"[*] Assembled sequence tensor of shape: {tuple(seq_tensor.shape)}")

    # 5. Initialize Model (Using untrained weights for this integration test)
    print("\n[PIPELINE STAGE 4] GRU Temporal Forecasting")
    model = TemporalForecaster(input_dim=len(feature_cols), forecast_horizons=4, num_classes=6)
    model.eval()

    # 6. Run Fast Gradient Attribution (XAI)
    print("\n[PIPELINE STAGE 5] Explainability & MITRE Mapping")
    explainer = FastGradientExplainer(feature_names=feature_cols)
    mapper = MitreMapper()
    
    # Explain the t+1 horizon forecast
    xai_results = explainer.explain_forecast(model, seq_tensor, horizon_idx=0)
    
    # Map the raw risk score to MITRE (Simulating a predicted class of '1' for Reconnaissance)
    # In a fully trained model, we would use torch.argmax(outputs['stage_trajectory'])
    mitre_results = mapper.evaluate_trajectory(
        predicted_class_idx=1, 
        risk_score=xai_results['forecasted_risk'], 
        current_features=df.iloc[-1].to_dict()
    )

    # 7. Generate Final Output JSON
    print("\n[PIPELINE STAGE 6] JSON Output Generation")
    
    final_output = {
        "event_id": f"sih-{int(time.time())}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_window": {
            "start": float(df.iloc[-1]['window_start']),
            "end": float(df.iloc[-1]['window_end'])
        },
        "forecast": {
            "horizon": "t+1 (+5 seconds)",
            "escalation_probability": mitre_results['escalation_probability'],
            "predicted_tactic": mitre_results['mapped_tactic'],
            "mitre_techniques": mitre_results['suggested_techniques'],
            "defense_engine_status": mitre_results['confidence_modifier']
        },
        "explainability": {
            "top_drivers": [
                {d['feature']: round(d['score'], 4)} for d in xai_results['top_positive_drivers']
            ]
        }
    }

    print("\n=========================================================")
    print(" 📡 FINAL SIH26153 SYSTEM OUTPUT (JSON)                  ")
    print("=========================================================")
    print(json.dumps(final_output, indent=4))
    print("=========================================================")
    print("[+] End-to-End Pipeline test completed successfully!")

if __name__ == "__main__":
    import time
    run_end_to_end()