"""Leaderboard, best-model selection and comparison plots (notebook cells 81-89)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from models.model_zoo import model_file_path  # noqa: E402


def build_leaderboard(results):
    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values(by=["F1-Macro", "ROC-AUC"], ascending=False).reset_index(drop=True)
    print("=== MODEL LEADERBOARD ===")
    print(leaderboard.to_string(float_format=lambda v: f"{v:.4f}"))
    leaderboard.to_csv(config.LEADERBOARD_PATH, index=False)
    print("Leaderboard saved at:", config.LEADERBOARD_PATH)
    return leaderboard


def identify_best_model(leaderboard):
    best_row = leaderboard.iloc[0]
    best_model_name, best_method = best_row["Model"], best_row["Method"]
    winning_model = joblib.load(model_file_path(best_model_name, best_method))
    print("=== BEST MODEL ===")
    print("Model:", best_model_name)
    print("Training Method:", best_method)
    for metric in ["F1-Macro", "F1-Fraud", "ROC-AUC", "PR-AUC"]:
        print(f"{metric}:", round(best_row[metric], 4))
    return best_model_name, best_method, winning_model


def plot_model_comparison(leaderboard, metric, filename):
    plt.figure(figsize=(12, 7))
    sns.barplot(data=leaderboard, x=metric, y="Model", hue="Method")
    plt.title(f"Model Comparison by {metric}")
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.legend(title="Training Method")
    plt.tight_layout()
    plt.savefig(config.EVAL_PLOTS_DIR / filename, dpi=150)
    plt.close()
