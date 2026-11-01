"""Metric calculations (notebook cells 99-103)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)


def final_metrics_table(y_true, y_pred_optimal, probabilities):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred_optimal),
        "Precision": precision_score(y_true, y_pred_optimal, zero_division=0),
        "Recall": recall_score(y_true, y_pred_optimal, zero_division=0),
        "F1-Fraud": f1_score(y_true, y_pred_optimal, zero_division=0),
        "F1-Macro": f1_score(y_true, y_pred_optimal, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
    }
    table = pd.DataFrame({"Metric": list(metrics), "Score": list(metrics.values())})
    return metrics, table


def financial_impact(test_df, y_pred_optimal):
    test_financial = test_df.copy()
    test_financial["Predicted_Fraud"] = y_pred_optimal
    total_fraud_amount = test_financial.loc[test_financial["Fraud_Flag"] == 1, "Transaction_Amount"].sum()
    fraud_prevented = test_financial.loc[
        (test_financial["Fraud_Flag"] == 1) & (test_financial["Predicted_Fraud"] == 1), "Transaction_Amount"].sum()
    fraud_missed = total_fraud_amount - fraud_prevented
    false_alarm_amount = test_financial.loc[
        (test_financial["Fraud_Flag"] == 0) & (test_financial["Predicted_Fraud"] == 1), "Transaction_Amount"].sum()
    fraud_capture_rate = fraud_prevented / total_fraud_amount * 100
    print(f"Total Actual Fraud Amount: Rs {total_fraud_amount:,.2f}")
    print(f"Fraud Amount Correctly Flagged: Rs {fraud_prevented:,.2f}")
    print(f"Fraud Amount Captured: {fraud_capture_rate:.2f}%")
    print(f"Fraud Amount Missed: Rs {fraud_missed:,.2f}")
    print(f"Value of False Alarms: Rs {false_alarm_amount:,.2f}")
    return fraud_capture_rate


def feature_importance(model, columns):
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_[0])
    else:
        return None
    df = pd.DataFrame({"Feature": columns, "Importance": importance_values})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)
