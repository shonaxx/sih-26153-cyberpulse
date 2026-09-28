import os
from pathlib import Path

def create_portable_repo():
    """Generates a cross-platform repository structure, skipping existing files."""
    
    # Define the base directory structure
    directories = [
        "configs",
        "data/raw_pcaps",
        "data/processed_csv",
        "data/scalers",
        "preprocessing",
        "state_builder",
        "models",
        "forecasting",
        "mitre",
        "explainability",
        "evaluation",
        "dashboard",
        "scripts",
        "tests"
    ]

    # Define initial files and their starter content
    files = {
        ".gitignore": (
            "__pycache__/\n"
            "*.pyc\n"
            ".DS_Store\n"
            ".venv/\n"  # Added your .venv here!
            "venv/\n"
            "data/raw_pcaps/*.pcap\n"
            "data/processed_csv/*.csv\n"
            "data/scalers/*.pkl\n"
            "models/saved_weights/*.pt\n"
        ),
        "README.md": (
            "# SIH26153: AI based Network Attack Forecasting\n\n"
            "A temporal sequence forecaster for mapping network behavior to MITRE ATT&CK stages.\n"
        ),
        
        # Configs
        "configs/default_config.yaml": (
            "window_size_sec: 10\n"
            "step_size_sec: 5\n"
            "sequence_length: 6\n"
            "forecast_horizons: 4\n"
        ),
        "configs/model_hyperparams.yaml": (
            "input_dim: 42\n"
            "hidden_dim: 64\n"
            "num_layers: 2\n"
            "dropout: 0.2\n"
            "learning_rate: 0.001\n"
        ),

        # Init files for module imports
        "preprocessing/__init__.py": "",
        "state_builder/__init__.py": "",
        "models/__init__.py": "",
        "forecasting/__init__.py": "",
        "mitre/__init__.py": "",
        "explainability/__init__.py": "",
        "evaluation/__init__.py": "",

        # Core Stubs 
        "preprocessing/pcap_parser.py": "# Streaming packet extractor using scapy.PcapReader\n",
        "preprocessing/flow_tracker.py": "# In-memory 5-tuple bidirectional flow tracking\n",
        "preprocessing/feature_extractor.py": "# Calculates entropy, IAT, and flag ratios\n",
        
        "state_builder/sliding_window.py": "# Aggregates features over sliding windows\n",
        "state_builder/normalizer.py": "# RobustScaler implementation for feature normalization\n",
        
        "models/baselines.py": "# Logistic Regression and Random Forest static classifiers\n",
        "models/gru_forecaster.py": "# PyTorch GRU Temporal Forecaster with Multi-Task Heads\n",
        "models/attention.py": "# Temporal Attention Layer\n",
        
        "forecasting/inference_engine.py": "# Manages forward passes and state sequence arrays\n",
        "forecasting/trajectory_evaluator.py": "# Converts logits to probability trajectories\n",
        
        "mitre/mapper.py": "# Maps telemetry features and risk scores to MITRE Tactics\n",
        "mitre/attack_specs.json": "{\n  \"TA0043\": \"Reconnaissance\",\n  \"TA0001\": \"Initial Access\"\n}\n",
        
        "explainability/gradient_attribution.py": "# Fast Integrated Gradients / Input x Gradient\n",
        "explainability/attention_weights.py": "# Extracts α weights from the temporal module\n",
        
        "evaluation/metrics.py": "# F1, Precision, Recall, FPR, Lead Time calculators\n",
        "evaluation/evaluate_horizons.py": "# Multi-step forecast degradation plotter\n",
        
        "dashboard/app.py": "# Streamlit main entrypoint\n",
        "dashboard/components.py": "# UI widgets (Risk gauges, timeline charts)\n",
        "dashboard/mock_stream.py": "# Local PCAP replay simulator for live dashboard testing\n",
        
        "scripts/train_baseline.py": "# Script to train and save Logistic Regression\n",
        "scripts/train_forecaster.py": "# Script to train the PyTorch GRU\n",
        "scripts/run_pipeline.py": "# End-to-end execution script (PCAP -> Dashboard)\n",
        
        "tests/test_pcap_parser.py": "# Pytest for packet extraction accuracy\n",
        "tests/test_sliding_window.py": "# Pytest for sequence boundary leakage prevention\n",
        "tests/test_inference.py": "# Pytest asserting CPU latency < 10ms\n"
    }

    print("Initializing SIH26153 Repository Structure...")

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        if dir_path.startswith("data/"):
            open(os.path.join(dir_path, ".gitkeep"), 'a').close()
        print(f"[+] Verified directory: {dir_path}")

    for file_path, content in files.items():
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # SAFEGUARD: Only write if the file does NOT exist
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[+] Created file: {file_path}")
        else:
            print(f"[~] Skipped existing file: {file_path}")

    print("\nRepository generated successfully! Your existing files were protected.")

if __name__ == "__main__":
    create_portable_repo()