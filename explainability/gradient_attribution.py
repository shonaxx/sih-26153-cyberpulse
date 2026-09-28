# Fast Integrated Gradients / Input x Gradient
import os
import sys
import torch
from typing import List, Dict, Any

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FastGradientExplainer:
    """
    Computes feature attribution for sequence models in < 5ms using Input x Gradient.
    Identifies which network features are driving the threat forecast up or down.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_forecast(self, model: torch.nn.Module, x_seq: torch.Tensor, horizon_idx: int = 0) -> Dict[str, Any]:
        """
        Calculates attribution for a specific forecast horizon.
        x_seq shape: (1, Sequence_Length, Features)
        """
        # 1. Prepare model and tensor for gradient tracking
        model.eval()
        model.zero_grad()
        
        # Clone tensor and enable gradient tracking
        x_input = x_seq.clone().detach().requires_grad_(True)
        
        # 2. Forward pass
        outputs = model(x_input)
        
        # Get the specific risk forecast we want to explain (e.g., t+1)
        # Output shape is (Batch, Horizons, 1). We want Batch 0, Horizon idx, Scalar 0
        target_risk = outputs['risk_trajectory'][0, horizon_idx, 0]
        
        # If the risk is very low, gradients might be near zero. 
        # We only strictly need explanations for elevated risks.
        
        # 3. Backward pass (The magic happens here)
        target_risk.backward()
        
        # 4. Calculate Input x Gradient Saliency
        # Saliency = Input Value * Gradient (How much the input moved the output)
        gradients = x_input.grad[0] # Extract batch 0 -> Shape: (Seq_Len, Features)
        inputs = x_input[0].detach() # Shape: (Seq_Len, Features)
        
        attribution_matrix = inputs * gradients
        
        # Sum the attributions across the time sequence to get overall feature importance
        # Shape becomes: (Features,)
        overall_feature_importance = attribution_matrix.sum(dim=0).cpu().numpy()
        
        # 5. Sort and categorize the features
        feature_scores = []
        for i, score in enumerate(overall_feature_importance):
            feature_scores.append({
                "feature": self.feature_names[i],
                "score": float(score),
                "direction": "up" if score > 0 else "down"
            })
            
        # Sort by absolute magnitude to find the most impactful features
        feature_scores.sort(key=lambda x: abs(x["score"]), reverse=True)
        
        # Separate into positive drivers (causing the alert) and negative drivers
        positive_drivers = [f for f in feature_scores if f["direction"] == "up"]
        negative_drivers = [f for f in feature_scores if f["direction"] == "down"]
        
        return {
            "forecasted_risk": float(target_risk.detach().cpu().numpy()),
            "top_positive_drivers": positive_drivers[:3], # Top 3 reasons FOR the alert
            "top_negative_drivers": negative_drivers[:2]  # Top 2 mitigating factors
        }

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    from models.gru_forecaster import TemporalForecaster
    
    print("--- Testing Fast Gradient Attribution (XAI) ---")
    
    # 1. Setup dummy model and data
    FEATURES = 42
    model = TemporalForecaster(input_dim=FEATURES, forecast_horizons=4, num_classes=6)
    
    # Generate 42 dummy feature names (e.g., f_0, f_1, ... f_41)
    # We will overwrite a few with realistic names so the test output looks real
    dummy_feature_names = [f"feature_{i}" for i in range(FEATURES)]
    dummy_feature_names[5] = "dst_port_entropy"
    dummy_feature_names[12] = "syn_rate"
    dummy_feature_names[25] = "mean_packet_length"
    
    # 1 Batch, 6 Windows (30s history), 42 Features
    dummy_input = torch.randn(1, 6, FEATURES) 
    
    # 2. Run Explainer
    explainer = FastGradientExplainer(feature_names=dummy_feature_names)
    
    import time
    start_time = time.time()
    
    # Let's explain the horizon at index 1 (t+2 / +10 seconds in the future)
    explanation = explainer.explain_forecast(model, dummy_input, horizon_idx=1)
    
    exec_time = (time.time() - start_time) * 1000
    
    print(f"\n[*] Execution Time: {exec_time:.2f} ms (Must be < 10ms for real-time dashboards)")
    print(f"[*] Forecasted Risk (t+2): {explanation['forecasted_risk']:.4f}\n")
    
    print("🔴 TOP ALERT DRIVERS (Pushed risk UP):")
    for d in explanation['top_positive_drivers']:
        print(f"   [+] {d['feature']}: {d['score']:.4f}")
        
    print("\n🟢 MITIGATING FACTORS (Pushed risk DOWN):")
    for d in explanation['top_negative_drivers']:
        print(f"   [-] {d['feature']}: {d['score']:.4f}")