"""Unit tests: input validation + model inference + feature engineering."""
import copy

import pytest

from src.features import ENGINEERED_FEATURES, engineer_features
from src.inference import (InvalidInputError, build_raw_frame, default_inputs, load_bundle,
                           predict, validate_inputs)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def risky_inputs():
    risky = default_inputs()
    risky.update({
        "Transaction_Amount": 95000.0, "Transaction_Hour": 2, "Account_Age_Days": 10, "KYC_Status": 0,
        "Device_Change_Flag": 1, "IP_Location_Match": 0, "Suspicious_Device": 1, "Rapid_Transaction_Flag": 1,
        "Login_Attempts": 6, "Transactions_Last_24h": 20, "Previous_Transaction_Count": 3,
        "Transaction_Type": "EMI",
    })
    return risky


# ------------------------------------------------------------ validation
def test_valid_inputs_pass_validation():
    assert validate_inputs(default_inputs()) is True


@pytest.mark.parametrize("field,value", [
    ("Transaction_Amount", -50.0),
    ("Transaction_Amount", 0),
    ("User_Age", 5),
    ("User_Age", 120),
    ("Transaction_Hour", 24),
    ("Login_Attempts", 0),
    ("Account_Age_Days", -1),
])
def test_invalid_numeric_inputs_are_rejected(field, value):
    bad = default_inputs()
    bad[field] = value
    with pytest.raises(InvalidInputError):
        validate_inputs(bad)


def test_missing_field_is_rejected():
    bad = default_inputs()
    del bad["Bank_Balance"]
    with pytest.raises(InvalidInputError):
        validate_inputs(bad)


def test_non_binary_flag_is_rejected():
    bad = default_inputs()
    bad["Suspicious_Device"] = 3
    with pytest.raises(InvalidInputError):
        validate_inputs(bad)


# ------------------------------------------------------- feature engineering
def test_feature_engineering_creates_all_engineered_columns(bundle):
    engineered = engineer_features(build_raw_frame(default_inputs()), bundle["stats"])
    for feature in ENGINEERED_FEATURES:
        assert feature in engineered.columns
    assert not engineered[ENGINEERED_FEATURES].isna().any().any()


# ------------------------------------------------------------- inference
def test_prediction_output_structure(bundle):
    result = predict(default_inputs(), bundle)
    assert set(result) == {"probability", "threshold", "label", "is_fraud"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in ("Fraud", "Legitimate")


def test_risky_transaction_scores_higher_than_normal(bundle):
    normal = predict(default_inputs(), bundle)["probability"]
    risky = predict(risky_inputs(), bundle)["probability"]
    assert risky > normal


def test_prediction_is_deterministic(bundle):
    first = predict(copy.deepcopy(default_inputs()), bundle)["probability"]
    second = predict(copy.deepcopy(default_inputs()), bundle)["probability"]
    assert first == second


def test_model_metadata_contains_threshold(bundle):
    assert 0.0 < bundle["meta"]["optimal_threshold"] < 1.0
    assert bundle["meta"]["best_model"]
