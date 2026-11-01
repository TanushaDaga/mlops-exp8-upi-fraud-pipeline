"""Inference helpers shared by the Streamlit app and the unit tests.

Raw user inputs -> same feature engineering as training -> fitted preprocessor
-> best model -> fraud probability -> decision at the tuned threshold.
"""
from src import config

import json
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.features import engineer_features
from src.preprocessing import COLUMNS_TO_DROP

SERVING_DIR = config.SERVING_DIR

KYC_MAP = {"Not Submitted": 0, "Pending": 1, "Verified": 2}

# (min, max) accepted for every numeric input
NUMERIC_LIMITS = {
    "Account_Age_Days": (0, 1825),
    "User_Age": (18, 64),
    "Transaction_Amount": (0.01, 10_000_000),
    "Transaction_Hour": (0, 23),
    "Previous_Transaction_Count": (0, 100_000),
    "Avg_Transaction_Value": (0, 10_000_000),
    "Transactions_Last_24h": (0, 1_000),
    "Login_Attempts": (1, 50),
    "Wallet_Balance": (0, 100_000_000),
    "Bank_Balance": (0, 1_000_000_000),
}
BINARY_FIELDS = ["Device_Change_Flag", "IP_Location_Match", "Suspicious_Device", "Rapid_Transaction_Flag"]


class InvalidInputError(ValueError):
    """Raised when the user supplied an impossible value."""


def load_bundle(serving_dir=SERVING_DIR):
    serving_dir = Path(serving_dir)
    meta = json.loads((serving_dir / "model_metadata.json").read_text(encoding="utf-8"))
    return {
        "model": joblib.load(serving_dir / "best_model.joblib"),
        "preprocessor": joblib.load(serving_dir / "preprocessor.joblib"),
        "stats": joblib.load(serving_dir / "reference_stats.joblib"),
        "meta": meta,
    }


def validate_inputs(inputs):
    for field, (low, high) in NUMERIC_LIMITS.items():
        if field not in inputs or inputs[field] is None:
            raise InvalidInputError(f"Missing value for {field}")
        value = inputs[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidInputError(f"{field} must be a number")
        if not (low <= value <= high):
            raise InvalidInputError(f"{field} must be between {low} and {high} (got {value})")
    for field in BINARY_FIELDS:
        if inputs.get(field) not in (0, 1):
            raise InvalidInputError(f"{field} must be 0 or 1")
    if inputs.get("KYC_Status") not in (0, 1, 2):
        raise InvalidInputError("KYC_Status must be 0, 1 or 2")
    return True


def build_raw_frame(inputs):
    """Turn the form values into the same raw schema the training data had."""
    txn_date = inputs.get("Transaction_Date") or date.today()
    row = dict(inputs)
    row["Transaction_Date"] = pd.to_datetime(txn_date)
    if row.get("Transaction_Type") != "Merchant":
        row["Merchant_Category"] = None  # structurally missing, like the training data
    frame = pd.DataFrame([row])
    for col in COLUMNS_TO_DROP:  # keep the schema identical to training
        if col not in frame.columns:
            frame[col] = None
    return frame


def predict(inputs, bundle):
    validate_inputs(inputs)
    raw = build_raw_frame(inputs)
    engineered = engineer_features(raw, bundle["stats"])
    X = engineered.drop(columns=[c for c in COLUMNS_TO_DROP if c in engineered.columns])
    X_processed = bundle["preprocessor"].transform(X)
    probability = float(bundle["model"].predict_proba(X_processed)[:, 1][0])
    threshold = bundle["meta"]["optimal_threshold"]
    is_fraud = probability >= threshold
    return {
        "probability": probability,
        "threshold": threshold,
        "label": "Fraud" if is_fraud else "Legitimate",
        "is_fraud": bool(is_fraud),
    }


def default_inputs():
    """A normal-looking legitimate transaction (used by tests and as UI defaults)."""
    return {
        "Account_Age_Days": 1200, "User_Age": 35, "KYC_Status": 2, "Transaction_Amount": 1500.0,
        "Transaction_Hour": 14, "Device_Change_Flag": 0, "Previous_Transaction_Count": 300,
        "Avg_Transaction_Value": 7000.0, "Transactions_Last_24h": 3, "Login_Attempts": 1,
        "IP_Location_Match": 1, "Suspicious_Device": 0, "Rapid_Transaction_Flag": 0,
        "Wallet_Balance": 20000.0, "Bank_Balance": 200000.0, "User_City": "Mumbai",
        "Transaction_Type": "P2P", "Merchant_Category": None, "Device_Type": "Android",
        "App_Version": "v5.3.0", "Transaction_Date": datetime(2024, 9, 11),
    }
