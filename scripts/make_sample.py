import numpy as np
import pandas as pd

SRC = "data/processed_csv/integration_scaled.csv"
DST = "data/sample/integration_sample.csv"
N = 5000

labels = pd.read_csv(SRC, usecols=["Label"])["Label"]
print("Total rows:", len(labels))
print(labels.value_counts().to_string())

is_att = (labels.to_numpy() != "BENIGN").astype(np.int64)
cs = np.concatenate([[0], np.cumsum(is_att)])
att = cs[N:] - cs[:-N]              # attack rows in every N-row window
score = np.minimum(att, N - att)    # most balanced window scores highest
start = int(np.argmax(score))
end = start + N

print("Best window starts at row", start, "with", int(att[start]), "attack rows")
parts, pos = [], 0
for chunk in pd.read_csv(SRC, chunksize=100_000):
    lo, hi = max(start - pos, 0), min(end - pos, len(chunk))
    if lo < hi:
        parts.append(chunk.iloc[lo:hi])
    pos += len(chunk)
    if pos >= end:
        break

sample = pd.concat(parts)
sample.to_csv(DST, index=False)
print("Saved", len(sample), "rows to", DST)
print(sample["Label"].value_counts().to_string())