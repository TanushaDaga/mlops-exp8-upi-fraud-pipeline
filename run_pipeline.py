"""Run the whole ML pipeline end-to-end:

    ETL + feature store -> EDA plots -> training (MLflow) -> evaluation -> SHAP

Usage:  python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("ETL + feature store", "src/pipeline.py"),
    ("EDA visualizations", "eda/eda_plots.py"),
    ("Model training + MLflow tracking", "models/train.py"),
    ("Evaluation + leaderboard", "evaluation/evaluate.py"),
    ("SHAP explainability", "shap/shap_explain.py"),
]

if __name__ == "__main__":
    for title, script in STAGES:
        print(f"\n{'#' * 70}\n# {title}  ({script})\n{'#' * 70}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print("\nPipeline finished. Launch the MLflow UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
