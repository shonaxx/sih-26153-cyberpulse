import os, sys
sys.path.append(os.getcwd())
import numpy as np, pandas as pd, torch
from models.gru_forecaster import TemporalForecaster

df = pd.read_csv("data/sample/integration_sample.csv")
cols = [c for c in df.columns if c != "Label"]
X = np.nan_to_num(df[cols].to_numpy(dtype=np.float32))
lab = df["Label"].to_numpy()
att = lab != "BENIGN"

m = TemporalForecaster(input_dim=X.shape[1])
m.load_state_dict(torch.load("models/saved_weights/gru_forecaster.pt", map_location="cpu"))
m.eval()

runs, s = [], 0
for i in range(1, len(lab) + 1):
    if i == len(lab) or lab[i] != lab[s]:
        runs.append((str(lab[s]), s, i - 1))
        s = i
print("first label runs (label, start, end):", runs[:15])

RUN, L = 20, 6
k = next((i for i in range(L + 1, len(att) - RUN)
        if att[i:i + RUN].all() and not att[i - 1]), None)
if k is None:
    raise SystemExit("no sustained attack run found")
print("sustained attack starts at row", k)
print("row | label at t+1 | risk t+1..t+4 | stage")
for i in range(k - 12, k + 9):
    with torch.no_grad():
        out = m(torch.from_numpy(X[i - L:i]).unsqueeze(0))
    r = out["risk_trajectory"].squeeze().numpy().round(2)
    s0 = int(out["stage_trajectory"][0, 0].argmax())
    print(i, lab[i], r, "stage", s0)