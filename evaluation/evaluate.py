"""Held-out evaluation for SIH26153: GRU vs persistence vs logistic regression.

Split: the file is cut into contiguous 5,000-row blocks; every 5th block is test.
Windows that cross a block boundary are dropped, so no window mixes train and test rows.
The GRU is retrained on the train blocks only and saved to a SEPARATE weights file,
so models/saved_weights/gru_forecaster.pt (used by the dashboard) is not touched.

Run from the repo root:   python evaluation/evaluate.py          (10 epochs)
Quick test:               python evaluation/evaluate.py 2
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

sys.path.append(os.getcwd())
from models.gru_forecaster import TemporalForecaster

DATA = "data/processed_csv/integration_scaled.csv"
OUT_WEIGHTS = "models/saved_weights/gru_eval_split.pt"
L, H, BLOCK, BS, SEED = 6, 4, 5000, 2048, 42
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LR_TRAIN_ROWS = 200_000

# Copy of the label -> stage mapping in scripts/train_forecaster.py (unmapped -> 0)
MAP = {
    "BENIGN": 0, "PortScan": 1,
    "FTP-Patator": 2, "SSH-Patator": 2,
    "Web Attack - Brute Force": 2, "Web Attack - Sql Injection": 2, "Web Attack - XSS": 2,
    "Infiltration": 3, "Bot": 4, "DoS Hulk": 4, "DoS GoldenEye": 4,
    "DoS slowloris": 4, "DoS Slowhttptest": 4, "DDoS": 4, "Heartbleed": 5,
}

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- load ----------------
head = pd.read_csv(DATA, nrows=0).columns
feat_cols = [c for c in head if c != "Label"]
df = pd.read_csv(DATA, dtype={c: np.float32 for c in feat_cols}, low_memory=False)
lab = df["Label"].map(MAP).fillna(0).astype(int).to_numpy()
X = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
del df
N, F = X.shape
print(f"Rows: {N:,}  Features: {F}  Attack rows: {(lab > 0).sum():,}")

# ---------------- blocked split ----------------
starts = np.arange(N - L - H)
blk = starts // BLOCK
ok = blk == (starts + L + H - 1) // BLOCK
is_test = blk % 5 == 4
train_idx = starts[ok & ~is_test]
test_idx = starts[ok & is_test]
print(f"Train windows: {len(train_idx):,}  Test windows: {len(test_idx):,}")

offs = np.arange(L)
fut_offs = L + np.arange(H)


def get_x(idx):
    return X[idx[:, None] + offs]                   # (B, L, F)


def get_future(idx):
    return lab[idx[:, None] + fut_offs]             # (B, H) stage ids


# ---------------- train GRU on train blocks only ----------------
model = TemporalForecaster(input_dim=F)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
bce, ce = nn.BCELoss(), nn.CrossEntropyLoss()
for ep in range(EPOCHS):
    t0 = time.time()
    model.train()
    perm = np.random.permutation(train_idx)
    total = 0.0
    for b in range(0, len(perm), BS):
        idx = perm[b:b + BS]
        x = torch.from_numpy(get_x(idx))
        fut = torch.from_numpy(get_future(idx)).long()
        out = model(x)
        risk = out["risk_trajectory"].squeeze(-1)
        loss = bce(risk, (fut > 0).float()) + 0.8 * ce(out["stage_trajectory"].reshape(-1, 6), fut.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {ep + 1}/{EPOCHS}  loss {total / max(1, len(perm) // BS):.4f}  ({time.time() - t0:.0f}s)")
os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
torch.save(model.state_dict(), OUT_WEIGHTS)

# ---------------- test truth ----------------
y = (get_future(test_idx) > 0).astype(int)                       # (n, H)
prev_attack = np.stack([lab[test_idx + j] > 0 for j in range(L)], 1).any(1)
onset = ~prev_attack                                             # window is fully benign

# GRU predictions
model.eval()
chunks = []
with torch.no_grad():
    for b in range(0, len(test_idx), 8192):
        x = torch.from_numpy(get_x(test_idx[b:b + 8192]))
        chunks.append(model(x)["risk_trajectory"].squeeze(-1).numpy())
p_gru = (np.concatenate(chunks) > 0.5).astype(int)

# Persistence: "an attack now means an attack at every future step"
cur = (lab[test_idx + L - 1] > 0).astype(int)
p_pers = np.repeat(cur[:, None], H, axis=1)


# Logistic regression on last row + window mean (clipped: raw values reach 1e8)
def lr_features(idx):
    xw = get_x(idx)
    return np.clip(np.concatenate([xw[:, -1, :], xw.mean(1)], axis=1), -10, 10)


rng = np.random.default_rng(SEED)
sub = rng.choice(train_idx, size=min(LR_TRAIN_ROWS, len(train_idx)), replace=False)
Xtr, Ytr = lr_features(sub), (get_future(sub) > 0).astype(int)
Xte = lr_features(test_idx)
p_lr = np.zeros_like(y)
for h in range(H):
    clf = LogisticRegression(max_iter=300)
    clf.fit(Xtr, Ytr[:, h])
    p_lr[:, h] = clf.predict(Xte)
    print(f"Logistic regression fitted for t+{h + 1}")


# ---------------- metrics ----------------
def scores(t, p):
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, fp / max(fp + tn, 1)


rows = []
for name, pred in [("Persistence", p_pers), ("Logistic regression", p_lr), ("GRU", p_gru)]:
    for h in range(H):
        prec, rec, f1, fpr = scores(y[:, h], pred[:, h])
        m = onset & (y[:, h] == 1)
        onset_rec = float(pred[m, h].mean()) if m.any() else float("nan")
        rows.append([name, f"t+{h + 1}", prec, rec, f1, fpr, onset_rec])

res = pd.DataFrame(rows, columns=["Model", "Horizon", "Precision", "Recall", "F1", "FPR",
                                "Onset recall"]).round(3)
os.makedirs("evaluation", exist_ok=True)
res.to_csv("evaluation/results.csv", index=False)

lines = ["| " + " | ".join(res.columns) + " |", "|" + "---|" * len(res.columns)]
for _, r in res.iterrows():
    lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
md = "\n".join(lines)
with open("evaluation/results.md", "w", encoding="utf-8") as fh:
    fh.write(md + "\n")

print("\nTest windows:", len(test_idx), "| attack rate at t+1:", round(float(y[:, 0].mean()), 3))
print("Windows that are fully benign but followed by an attack (onset windows) at t+1:",
    int((onset & (y[:, 0] == 1)).sum()))
print("\n" + md)
print("\nSaved evaluation/results.csv, evaluation/results.md and", OUT_WEIGHTS)