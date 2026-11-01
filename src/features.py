"""Feature engineering (notebook cells 37-45).

All statistics (medians, modes, caps, IQR bounds) are learned from the TRAINING
split only.  ``compute_reference_stats`` captures them once so the very same
numbers can be re-used at inference time by the Streamlit service.
"""
from src import config  # noqa: F401

import numpy as np

NUMERIC_IMPUTATION_COLUMNS = [
    "User_Age", "Avg_Transaction_Value", "Login_Attempts", "Wallet_Balance",
    "Bank_Balance", "Account_Age_Days", "Previous_Transaction_Count", "Transactions_Last_24h",
]
CATEGORICAL_IMPUTATION_COLUMNS = ["User_City", "Transaction_Type", "Device_Type", "App_Version"]
IQR_FLAG_COLUMNS = ["Transaction_Amount", "Avg_Transaction_Value", "Login_Attempts"]

ENGINEERED_FEATURES = [
    "Amount_to_Avg_Ratio", "Amount_to_Bank_Ratio", "Total_Liquidity", "Txn_to_Liquidity_Ratio",
    "Txn_Velocity_per_Day", "Is_Night", "Is_Night_Transaction", "Is_New_Account", "Is_Weekend",
    "High_Value_Night_Txn", "Low_History_High_Value", "Multi_Flag_Count", "Security_Risk_Score",
    "IP_Device_Mismatch", "Suspicious_Multi_Flag", "KYC_Risk_Flag", "Log_Transaction_Amount",
    "Log_Avg_Transaction_Value",
]


def compute_reference_stats(train_df):
    """Calculate imputation values and training-only limits."""
    numeric_medians = {c: train_df[c].median() for c in NUMERIC_IMPUTATION_COLUMNS}
    categorical_modes = {c: train_df[c].mode(dropna=True).iloc[0] for c in CATEGORICAL_IMPUTATION_COLUMNS}
    # Merchant category needs special treatment
    merchant_mode = (
        train_df.loc[train_df["Transaction_Type"] == "Merchant", "Merchant_Category"]
        .dropna().mode().iloc[0]
    )
    iqr_bounds = {}
    for col in IQR_FLAG_COLUMNS:
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1
        iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return {
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
        "merchant_mode": merchant_mode,
        "avg_value_cap": train_df["Avg_Transaction_Value"].quantile(0.99),
        "login_cap": train_df["Login_Attempts"].quantile(0.99),
        "iqr_bounds": iqr_bounds,
        "train_amount_median": train_df["Transaction_Amount"].median(),
    }


def engineer_features(data, stats):
    """Complete feature engineering function (same logic as the notebook)."""
    data = data.copy()

    # 1. NUMERICAL IMPUTATION
    for col in NUMERIC_IMPUTATION_COLUMNS:
        data[col] = data[col].fillna(stats["numeric_medians"][col])

    # 2. CATEGORICAL IMPUTATION
    for col in CATEGORICAL_IMPUTATION_COLUMNS:
        data[col] = data[col].fillna(stats["categorical_modes"][col])
    # Merchant_Category is structurally missing for non-Merchant transactions.
    data.loc[(data["Merchant_Category"].isna()) & (data["Transaction_Type"] != "Merchant"), "Merchant_Category"] = "Not_Applicable"
    # Remaining missing Merchant_Category values belong to Merchant transactions.
    data["Merchant_Category"] = data["Merchant_Category"].fillna(stats["merchant_mode"])

    # 3. OUTLIER TREATMENT (training-only 99th percentile caps)
    data["Avg_Transaction_Value"] = data["Avg_Transaction_Value"].clip(upper=stats["avg_value_cap"])
    data["Login_Attempts"] = data["Login_Attempts"].clip(upper=stats["login_cap"])
    # IMPORTANT: Transaction_Amount is NOT capped because high amounts are a fraud signal.

    # 4. TRAINING-BASED OUTLIER FLAGS
    for col in IQR_FLAG_COLUMNS:
        lower, upper = stats["iqr_bounds"][col]
        data[f"{col}_IQR_Outlier"] = ((data[col] < lower) | (data[col] > upper)).astype(int)

    # 5. DATE/TIME FEATURES
    data["Transaction_Month"] = data["Transaction_Date"].dt.month
    data["Transaction_DayOfWeek"] = data["Transaction_Date"].dt.dayofweek
    data["Is_Weekend"] = data["Transaction_DayOfWeek"].isin([5, 6]).astype(int)
    data["Is_Night"] = data["Transaction_Hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype(int)
    data["Is_Night_Transaction"] = data["Is_Night"]

    # 6. ACCOUNT FEATURES
    data["Is_New_Account"] = (data["Account_Age_Days"] < 30).astype(int)
    data["KYC_Risk_Flag"] = (data["KYC_Status"] < 2).astype(int)

    # 7. FINANCIAL FEATURES
    data["Amount_to_Avg_Ratio"] = data["Transaction_Amount"] / (data["Avg_Transaction_Value"] + 1e-5)
    data["Amount_to_Bank_Ratio"] = data["Transaction_Amount"] / (data["Bank_Balance"] + 1e-5)
    data["Total_Liquidity"] = data["Wallet_Balance"] + data["Bank_Balance"]
    data["Txn_to_Liquidity_Ratio"] = data["Transaction_Amount"] / (data["Total_Liquidity"] + 1e-5)

    # 8. BEHAVIORAL FEATURES
    data["Txn_Velocity_per_Day"] = data["Previous_Transaction_Count"] / (data["Account_Age_Days"] + 1)
    data["High_Value_Night_Txn"] = (
        (data["Transaction_Hour"] < 5) & (data["Transaction_Amount"] > stats["train_amount_median"])
    ).astype(int)
    data["Low_History_High_Value"] = (
        (data["Previous_Transaction_Count"] < 10) & (data["Amount_to_Avg_Ratio"] > 3)
    ).astype(int)

    # 9. SECURITY FEATURES
    data["IP_Device_Mismatch"] = ((data["IP_Location_Match"] == 0) & (data["Device_Change_Flag"] == 1)).astype(int)
    data["Multi_Flag_Count"] = (
        (data["IP_Location_Match"] == 0).astype(int)
        + data["Device_Change_Flag"].astype(int)
        + data["Suspicious_Device"].astype(int)
        + (data["Login_Attempts"] >= 4).astype(int)
    )
    data["Security_Risk_Score"] = (
        data["Suspicious_Device"].astype(int)
        + data["Rapid_Transaction_Flag"].astype(int)
        + (1 - data["IP_Location_Match"].astype(int))
        + data["Device_Change_Flag"].astype(int)
    )
    data["Suspicious_Multi_Flag"] = (data["Multi_Flag_Count"] >= 2).astype(int)

    # 10. LOG TRANSFORMATIONS
    data["Log_Transaction_Amount"] = np.log1p(data["Transaction_Amount"])
    data["Log_Avg_Transaction_Value"] = np.log1p(data["Avg_Transaction_Value"])
    return data
