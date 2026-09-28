import os, sys
sys.path.append(os.getcwd())
import numpy as np, pandas as pd, torch
from models.gru_forecaster import TemporalForecaster

df = pd.read_csv("data/sample/integration_sample.csv")
cols = [c for c in df.columns if c != "Label"]
X = np.nan_to_num(df[cols].to_numpy(dtype=np.float32))

sd = torch.load("models/saved_weights/gru_forecaster.pt", map_location="cpu")
m = TemporalForecaster(input_dim=X.shape[1])
m.load_state_dict(sd)
m.eval()

L = 6
idx = np.arange(L, len(X), 5)
xs = torch.from_numpy(np.stack([X[i - L:i] for i in idx]))
with torch.no_grad():
    out = m(xs)
risk = out["risk_trajectory"].squeeze(-1).numpy()
stage = out["stage_trajectory"].argmax(-1).numpy()
lab = df["Label"].to_numpy()[idx]

for name in np.unique(lab):
    k = lab == name
    print(name, "| windows:", int(k.sum()),
        "| mean risk t+1..t+4:", risk[k].mean(0).round(3),
        "| stage counts (t+1):", np.bincount(stage[k, 0], minlength=6))
print("input min / max:", X.min(), X.max())