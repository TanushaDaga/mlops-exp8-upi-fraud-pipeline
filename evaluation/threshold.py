"""Threshold optimisation, classification reports, confusion matrix (cells 91-97)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve  # noqa: E402


def optimize_threshold(y_true, probabilities):
    precision_values, recall_values, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = 2 * precision_values[:-1] * recall_values[:-1] / (precision_values[:-1] + recall_values[:-1] + 1e-10)
    best_threshold_index = np.argmax(f1_values)
    optimal_threshold = thresholds[best_threshold_index]
    print("Default threshold:", config.DEFAULT_THRESHOLD)
    print("Optimal threshold:", round(optimal_threshold, 4))
    print("Best F1 at optimal threshold:", round(f1_values[best_threshold_index], 4))
    return thresholds, f1_values, optimal_threshold


def plot_threshold(thresholds, f1_values, optimal_threshold, best_model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_values)
    plt.axvline(0.50, linestyle="--", label="Default Threshold = 0.50")
    plt.axvline(optimal_threshold, linestyle="--", label=f"Optimal = {optimal_threshold:.3f}")
    plt.title(f"Threshold Optimization - {best_model_name}")
    plt.xlabel("Probability Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(config.EVAL_PLOTS_DIR / "threshold_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()


def report_both_thresholds(y_true, probabilities, optimal_threshold):
    y_pred_default = (probabilities >= config.DEFAULT_THRESHOLD).astype(int)
    y_pred_optimal = (probabilities >= optimal_threshold).astype(int)
    print("=== DEFAULT THRESHOLD (0.50) ===")
    print(classification_report(y_true, y_pred_default, target_names=["Legitimate", "Fraud"], zero_division=0))
    print("\n=== OPTIMAL THRESHOLD ===")
    print(classification_report(y_true, y_pred_optimal, target_names=["Legitimate", "Fraud"], zero_division=0))
    return y_pred_default, y_pred_optimal


def plot_confusion_matrix(y_true, y_pred_optimal, best_model_name):
    cm = confusion_matrix(y_true, y_pred_optimal)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(config.EVAL_PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    return cm
