"""Extract, convert, validate and split the UPI fraud dataset (notebook cells 5-35)."""
from src import config  # noqa: F401  (must be first: thread settings)

import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_COLUMNS = [
    "Account_Age_Days", "User_Age", "Transaction_Amount", "Transaction_Hour",
    "Device_Change_Flag", "Previous_Transaction_Count", "Avg_Transaction_Value",
    "Transactions_Last_24h", "Login_Attempts", "Wallet_Balance", "Bank_Balance",
]
OUTLIER_COLUMNS = ["Transaction_Amount", "Avg_Transaction_Value", "Login_Attempts"]


def extract_raw(file_path=config.RAW_DATA_PATH, sheet_name=config.SHEET_NAME):
    """Extract: load the Excel sheet; row index 3 holds the real column names."""
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    print("Raw Excel shape:", raw_df.shape)
    column_names = raw_df.iloc[3].astype(str).str.strip().str.replace(" ", "_", regex=False)
    df = raw_df.iloc[4:].copy()
    df.columns = column_names
    df = df.reset_index(drop=True)
    df.columns.name = None
    print("Clean extracted shape:", df.shape)
    return df


def inspect_dataset(df):
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("\nColumn names:")
    for i, col in enumerate(df.columns, start=1):
        print(i, ":", col)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nDataset information:")
    df.info()


def missing_value_summary(df):
    summary = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Total_Rows": len(df),
        "Missing_Percentage": (df.isna().sum() / len(df) * 100).round(2),
    })
    return summary.sort_values("Missing_Count", ascending=False)


def convert_types(df):
    """Data types and conversion."""
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Transaction_Time"] = df["Transaction_Time"].astype("string").str.strip()
    return df


def validate_data(df):
    parsed_hour = pd.to_datetime(df["Transaction_Time"], format="%H:%M", errors="coerce").dt.hour
    report = {
        "Duplicate complete rows": int(df.duplicated().sum()),
        "Duplicate Transaction IDs": int(df["Transaction_ID"].duplicated().sum()),
        "Invalid Transaction Amounts": int((df["Transaction_Amount"] <= 0).sum()),
        "Invalid Account Ages": int(((df["Account_Age_Days"] < 0) | (df["Account_Age_Days"] > 1825)).sum()),
        "Invalid User Ages": int(((df["User_Age"] < 18) | (df["User_Age"] > 64)).sum()),
        "Invalid Transaction Hours": int(((df["Transaction_Hour"] < 0) | (df["Transaction_Hour"] > 23)).sum()),
        "Transaction hour mismatches": int((parsed_hour.notna() & (parsed_hour != df["Transaction_Hour"])).sum()),
    }
    for name, value in report.items():
        print(f"{name}:", value)
    print("\nTarget values:")
    print(df["Fraud_Flag"].value_counts(dropna=False))
    return report


def encode_target_and_binaries(df, verbose=True):
    """Convert target and binary security variables."""
    df["Fraud_Flag"] = df["Fraud_Flag"].map({"No": 0, "Yes": 1})
    binary_mapping = {"No": 0, "Yes": 1}
    for col in ["IP_Location_Match", "Suspicious_Device", "Rapid_Transaction_Flag"]:
        df[col] = df[col].map(binary_mapping)
    df["Device_Change_Flag"] = pd.to_numeric(df["Device_Change_Flag"], errors="coerce")
    df["KYC_Status"] = df["KYC_Status"].map({"Not Submitted": 0, "Pending": 1, "Verified": 2})
    print("Target distribution:")
    print(df["Fraud_Flag"].value_counts())
    print("\nKYC encoding:")
    print(df["KYC_Status"].value_counts())
    return df


def outlier_summary(df):
    """Outlier detection using IQR."""
    results = []
    for col in OUTLIER_COLUMNS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        results.append({
            "Column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower_Bound": lower_bound, "Upper_Bound": upper_bound,
            "Outlier_Count": mask.sum(), "Outlier_Percentage": mask.mean() * 100,
        })
    return pd.DataFrame(results)


def amount_outlier_fraud_rates(df):
    """Are Transaction_Amount outliers fraud signals?"""
    q1 = df["Transaction_Amount"].quantile(0.25)
    q3 = df["Transaction_Amount"].quantile(0.75)
    iqr = q3 - q1
    amount_upper = q3 + 1.5 * iqr
    amount_outlier = df["Transaction_Amount"] > amount_upper
    print("Transaction amount IQR upper bound:")
    print(f"Rs {amount_upper:,.2f}")
    print("\nFraud rate among amount outliers:")
    print(df.loc[amount_outlier, "Fraud_Flag"].mean() * 100, "%")
    print("\nFraud rate among non-outliers:")
    print(df.loc[~amount_outlier, "Fraud_Flag"].mean() * 100, "%")


def load_clean_dataframe(verbose=False):
    """Extract -> convert -> validate -> encode (the dataframe EDA / split use)."""
    df = extract_raw()
    df = convert_types(df)
    if verbose:
        validate_data(df)
    return encode_target_and_binaries(df, verbose=verbose)


def split_data(df):
    """80/20 stratified split."""
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["Fraud_Flag"]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)
    print("\nTraining fraud rate:")
    print(train_df["Fraud_Flag"].mean() * 100)
    print("\nTesting fraud rate:")
    print(test_df["Fraud_Flag"].mean() * 100)
    return train_df, test_df
