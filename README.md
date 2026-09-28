# CyberPulse: Next-Record Attack Risk Detection with a GRU and a SOC Dashboard

**Smart India Hackathon 2026 submission (Problem Statement SIH26153)**

CyberPulse trains a PyTorch GRU on CIC-IDS-2017 flow records. For each window of 6 consecutive flow records it outputs an attack-risk probability and a MITRE-style stage prediction for each of the next 4 records. A Streamlit dashboard replays the data through the model and explains each prediction with Input x Gradient attributions.

## What this project is, and is not

- **It is** a next-record attack-risk detector with a working dashboard and per-feature explanations.
- **It is not** a demonstrated early-warning system. On a held-out evaluation, the model matches a trivial "persistence" baseline and does not flag attacks before they begin (onset recall of at most 0.5%). See [Evaluation](#evaluation).
- **The horizons t+1 to t+4 are the next 1 to 4 rows of the dataset**, where each row is one flow record. They are not seconds or minutes of future time.

## Features

- Multi-horizon output: attack risk and stage for the next 4 flow records.
- Explainability: Input x Gradient attribution for the t+1 risk output, shown per feature in the dashboard.
- Streamlit and Plotly dashboard that replays rows from a CSV and shows the predicted risk next to the actual label of the next row.
- Comparison against persistence and logistic-regression baselines, with an onset-recall metric (`evaluation/evaluate.py`).

## Project structure

```text
configs/         model hyperparameters (input_dim 78, hidden 64, 2 layers, dropout 0.2)
dashboard/       Streamlit app (app.py)
data/            dataset chunks, small sample CSV (integration_sample.csv)
evaluation/      evaluate.py (held-out comparison), results.csv, results.md
explainability/  gradient attribution, attention weights
forecasting/     inference engine, trajectory evaluator
mitre/           attack-stage specs and mapper
models/          GRU forecaster, baselines, saved_weights/gru_forecaster.pt
preprocessing/   PCAP parsing, flow tracking, feature extraction, CSV merging
scripts/         train_forecaster.py, train_baseline.py, check_onset.py, ...
state_builder/   sliding-window builder, RobustScaler normalizer
tests/
```

## Setup and running the dashboard

Requirements: Python 3.9 or higher. A GPU is optional; the code falls back to CPU.

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Run this from the repository root. The dashboard loads `models/saved_weights/gru_forecaster.pt` and replays `data/sample/integration_sample.csv`, which is committed to the repo. If the full `data/processed_csv/integration_scaled.csv` exists locally, the dashboard can use it too.

**Which weights the dashboard uses.** `gru_forecaster.pt` was trained by `scripts/train_forecaster.py` on all rows, with no held-out split. The held-out numbers below come from a separate retrain (`gru_eval_split.pt`, not committed) that uses the same architecture and training recipe. The table therefore describes this model design on unseen data, not the exact dashboard weights.

## Model and training

- Input: 6 consecutive rows, 78 features each (scaled with `RobustScaler`).
- Architecture: 2-layer GRU, hidden size 64, dropout 0.2, with a risk head (4 outputs) and a stage head (6 classes per horizon).
- Loss: `BCE(risk) + 0.8 * CrossEntropy(stage)`. Adam, learning rate 1e-3, batch size 2048, 10 epochs.
- Stage mapping: 0 benign, 1 PortScan, 2 brute force (FTP/SSH-Patator, web brute force), 3 Infiltration, 4 Bot and DoS/DDoS, 5 Heartbleed.
- Memory: `scripts/train_forecaster.py` builds all windows in memory (about 3 GB for the window tensor, more at peak), so 8 GB of RAM or more is recommended.

## Dataset

The raw CIC-IDS-2017 subset in `data/dataset_chunks/` has 1,425,079 rows and 78 features:

| Label | Rows |
|---|---|
| BENIGN | 999,642 |
| DoS Hulk | 231,073 |
| PortScan | 158,930 |
| DoS GoldenEye | 10,293 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Heartbleed | 11 |

This subset has no Infiltration, Bot, Web Attack or DDoS rows, so stage 3 never occurs and stage 5 has only 11 rows. **The stage head is therefore not meaningfully evaluated for those stages.** The training and evaluation file (`integration_scaled.csv`) has 1,423,147 rows and is not committed because of its size. The preprocessing steps that turn the raw chunks into `integration_scaled.csv` (cleaning, then scaling with `RobustScaler`) are not documented in this README yet.

To load the chunks:

```python
import glob
import pandas as pd

files = sorted(glob.glob("data/dataset_chunks/part_*.csv"))
df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
df.columns = df.columns.str.strip()   # raw column names have leading spaces, e.g. " Label"
```

## Evaluation

`python evaluation/evaluate.py` (add `2` for a quick 2-epoch test) retrains the GRU on part of the data and compares it with two baselines on held-out windows.

### Setup

- **Windows:** 6 input rows, targets are the labels of the next 4 rows. One row is one flow record.
- **Split:** the file is cut into contiguous 5,000-row blocks and every 5th block is held out for testing (20%). Windows that cross a block boundary are dropped, so no window mixes train and test rows. This gives 1,137,948 train and 282,633 test windows.
- **Risk target:** an attack (any non-benign label) at horizon t+h.
- **Persistence baseline:** predicts that the last observed row's state (attack or benign) holds at every future step.
- **Logistic regression baseline:** trained on 200,000 randomly sampled training windows, with the last row and the window mean as features (clipped to [-10, 10]), `max_iter=300`.
- **GRU:** trained on all training windows, with a 0.5 decision threshold.
- **Onset recall:** among windows whose 6 input rows are all benign, the fraction whose upcoming attack the model flags. There are 2,650 such windows at t+1.

### Results (held-out test windows, attack rate 0.313 at t+1)

| Model | Horizon | Precision | Recall | F1 | FPR | Onset recall |
|---|---|---|---|---|---|---|
| Persistence | t+1 | 0.936 | 0.936 | 0.936 | 0.029 | 0.0 |
| Persistence | t+2 | 0.927 | 0.927 | 0.927 | 0.033 | 0.0 |
| Persistence | t+3 | 0.924 | 0.924 | 0.924 | 0.035 | 0.0 |
| Persistence | t+4 | 0.921 | 0.921 | 0.921 | 0.036 | 0.0 |
| Logistic regression | t+1 | 0.959 | 0.93 | 0.944 | 0.018 | 0.002 |
| Logistic regression | t+2 | 0.957 | 0.926 | 0.941 | 0.019 | 0.003 |
| Logistic regression | t+3 | 0.958 | 0.923 | 0.94 | 0.018 | 0.004 |
| Logistic regression | t+4 | 0.957 | 0.923 | 0.94 | 0.019 | 0.005 |
| GRU | t+1 | 0.925 | 0.939 | 0.932 | 0.035 | 0.002 |
| GRU | t+2 | 0.923 | 0.929 | 0.926 | 0.036 | 0.003 |
| GRU | t+3 | 0.923 | 0.926 | 0.925 | 0.035 | 0.005 |
| GRU | t+4 | 0.924 | 0.925 | 0.924 | 0.035 | 0.005 |

### Findings

- The GRU performs on par with persistence (F1 0.924 to 0.932 vs 0.921 to 0.936) and slightly below logistic regression (0.940 to 0.944).
- The GRU's false-positive rate (0.035 to 0.036) is about twice logistic regression's (0.018 to 0.019).
- Onset recall is at most 0.5% for every model, so the evaluation shows no early-warning ability at the 0.5 threshold.
- Persistence scores well because attacks are sticky in this dataset: on the raw chunks, 93.5% of attack rows are followed by another attack row, while only 2.7% of benign rows are followed by an attack row. Most of every model's F1 comes from continuing an ongoing attack.

### Limitations

- The split is blocked but not chronological. Train and test blocks are interleaved, and long attack runs span many blocks, so the same attack episodes appear on both sides. The results measure generalization to unseen segments of familiar traffic, not to future time or to new attack types.
- `RobustScaler` was fit on the full file before splitting (a minor leak).
- Onset windows are rare (about 0.9% of test windows), and results use a fixed 0.5 threshold. Threshold-free metrics such as AUC on onset windows were not computed.
- The logistic regression baseline was trained on a 200,000-window subsample with `max_iter=300` and was not tuned. The GRU still did not beat it.
- The stage head is not evaluated (see Dataset).
- Onset recall of 0.002 to 0.005 on 2,650 windows means only about 5 to 13 windows were caught, so differences between the GRU and logistic regression there are within noise.
- Results come from a single run with seed 42.

### Future work

- Build real time-based windows with the existing `SlidingWindowBuilder`, so t+1 to t+4 mean seconds instead of rows.
- Evaluate with a chronological split and a gap, and hold out entire attack types.
- Report threshold-free onset metrics and tune the baselines fairly.
