# PyTorch GRU Temporal Forecaster with Multi-Task Heads
import os
import sys
import torch
import torch.nn as nn

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TemporalForecaster(nn.Module):
    """
    Direct multi-horizon GRU forecaster for SIH26153.
    Uses a Unidirectional GRU to learn sequence transitions and directly 
    forecasts multi-step threat trajectories using one head per horizon.
    """
    def __init__(self, input_dim=42, hidden_dim=64, num_layers=2, dropout=0.2, forecast_horizons=4, num_classes=6):
        super(TemporalForecaster, self).__init__()
        
        self.forecast_horizons = forecast_horizons
        
        # 1. Feature Embedding Layer
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Temporal Sequence Backbone (MUST be bidirectional=False to prevent future leakage)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False 
        )
        
        # 3. Direct Multi-Horizon Projection Heads
        # We create separate layers for each future time step (t+1, t+2, t+3, t+4)
        
        # Risk Heads: Outputs a single value [0, 1] for each horizon
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid() # Squashes output between 0% and 100%
            ) for _ in range(forecast_horizons)
        ])
        
        # Stage Heads: Outputs logits across 'C' MITRE classes for each horizon
        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
                # No Softmax here, as PyTorch CrossEntropyLoss applies it internally
            ) for _ in range(forecast_horizons)
        ])

    def forward(self, x):
        """
        x shape expected: (Batch, Sequence_Length, Features) e.g., (32, 6, 42)
        """
        # Embed the raw features
        x_emb = torch.relu(self.feature_proj(x))
        
        # Pass through the temporal GRU
        gru_out, hidden_state = self.gru(x_emb)
        
        # Extract the final context state (the model's memory at current time 't')
        # gru_out shape is (Batch, Seq_Len, Hidden_Dim). We want the last time step.
        context_vector = gru_out[:, -1, :] 
        
        # Initialize lists to hold our future predictions
        risk_forecasts = []
        stage_forecasts = []
        
        # Project into the future (K steps)
        for k in range(self.forecast_horizons):
            risk_k = self.risk_heads[k](context_vector)
            stage_k = self.stage_heads[k](context_vector)
            
            risk_forecasts.append(risk_k)
            stage_forecasts.append(stage_k)
            
        # Stack the outputs into neat tensors
        # Risk Output Shape: (Batch, Horizons, 1)
        risk_trajectory = torch.stack(risk_forecasts, dim=1)
        
        # Stage Output Shape: (Batch, Horizons, Num_Classes)
        stage_trajectory = torch.stack(stage_forecasts, dim=1)
        
        return {
            "risk_trajectory": risk_trajectory,
            "stage_trajectory": stage_trajectory
        }

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing PyTorch GRU Forecaster Architecture ---")
    
    # Define our dimensions based on SIH design
    BATCH_SIZE = 4      # Simulating 4 network segments simultaneously
    SEQ_LENGTH = 6      # 6 consecutive rows of history
    FEATURES = 42       # Our extracted state vector size
    HORIZONS = 4        # Predicting t+1, t+2, t+3, t+4
    MITRE_STAGES = 6    # 0=Normal, 1=Recon, 2=Brute force/Web, 3=Infiltration, 4=DoS/Botnet, 5=Heartbleed
    
    print(f"[*] Initializing Model on CPU...")
    model = TemporalForecaster(
        input_dim=FEATURES, 
        forecast_horizons=HORIZONS, 
        num_classes=MITRE_STAGES
    )
    
    print(f"[*] Generating dummy sequence tensor (Batch={BATCH_SIZE}, SeqLen={SEQ_LENGTH}, Feat={FEATURES})...")
    # torch.randn generates random decimal numbers simulating our scaled CSV data
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, FEATURES)
    
    print("[*] Running Forward Pass...")
    outputs = model(dummy_input)
    
    print("\n--- Output Tensor Shapes ---")
    print(f"Expected Risk Shape:  ({BATCH_SIZE}, {HORIZONS}, 1)")
    print(f"Actual Risk Shape:    {tuple(outputs['risk_trajectory'].shape)}")
    
    print(f"\nExpected Stage Shape: ({BATCH_SIZE}, {HORIZONS}, {MITRE_STAGES})")
    print(f"Actual Stage Shape:   {tuple(outputs['stage_trajectory'].shape)}")
    
    print("\n[+] Model forward pass successful! Tensors flow without dimension errors.")

